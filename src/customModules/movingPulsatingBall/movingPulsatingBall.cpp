/*
* MPBM State Effector Implementation (Robust Version)
* Features:
* - Soft Penalty with Tuned Parameters (Prevents Numerical Explosion)
* - Friction Force Clamping (Prevents Singularity at small L)
* - Internal State Logging (r, v, T_Li) for Validation
*/

#include "movingPulsatingBall.h"
#include "architecture/utilities/linearAlgebra.h"
#include "architecture/utilities/avsEigenSupport.h"
#include <iostream>
#include <cmath>

Eigen::Matrix3d eigenTilde(Eigen::Vector3d v) {
   Eigen::Matrix3d m;
   m << 0, -v[2], v[1],
        v[2], 0, -v[0],
        -v[1], v[0], 0;
   return m;
}

MovingPulsatingBall::MovingPulsatingBall() {
   // --- 500kg Spacecraft / 100kg Fuel Setup ---
   this->massInit = 100.0;            
   this->radiusTank = 0.50;           
   this->radiusSlugMin = 0.10;        
   this->kinematicViscosity = 2.839e-6; // Hydrazine
   this->surfaceTension = 0.066;      
   this->rho = 1004.0;                
   this->t_sr = 1.0;                  
   
   // [Fix 1] Numerical Stability Tuning
   // Lower stiffness to prevent explosion at dt=0.01s or 0.001s
   // Higher damping to make it "sticky" rather than "bouncy"
   this->k_barrier = 1000.0;  
   this->c_barrier = 100.0;   

   this->r_TB_B.setZero(); 

   // [Init] Default Initial States (Can be overwritten by Python)
   this->r_Init_B << 0.0, 0.0, 0.1; 
   this->v_Init_B.setZero();

   this->nameOfPosState = "mpbmPos";
   this->nameOfVelState = "mpbmVel";
   this->nameOfOmegaState = "mpbmOmega";
}

MovingPulsatingBall::~MovingPulsatingBall() {}

void MovingPulsatingBall::Reset(uint64_t CurrentSimNanos) {
   this->effProps.mEff = this->massInit;
}

void MovingPulsatingBall::registerStates(DynParamManager& states) {
   Eigen::Vector3d initOmega; initOmega.setZero();

   // [Init] Use Configured Initial States
   this->posState = states.registerState(3, 1, this->nameOfPosState);
   this->posState->setState(this->r_Init_B);

   this->velState = states.registerState(3, 1, this->nameOfVelState);
   this->velState->setState(this->v_Init_B);

   this->omegaState = states.registerState(3, 1, this->nameOfOmegaState);
   this->omegaState->setState(initOmega);
}

void MovingPulsatingBall::linkInStates(DynParamManager& states) {
}

void MovingPulsatingBall::updateEffectorMassProps(double integTime) {
   Eigen::Vector3d r_rel = this->posState->getState();
   Eigen::Vector3d v_rel = this->velState->getState();
   
   double r_norm = r_rel.norm();
   // Safety clamp for calculation only
   if(r_norm > this->radiusTank) r_norm = this->radiusTank - 1e-4;
   
   this->currentSlugRadius = this->radiusTank - r_norm; 
   
   this->r_SB_B = this->r_TB_B + r_rel; 
   this->v_SB_B = v_rel;
   
   this->effProps.mEff = this->massInit;
   this->effProps.rEff_CB_B = this->r_SB_B;
   this->effProps.rEffPrime_CB_B = this->v_SB_B;
   
   // Inertia I_s = 0.4 * m * L^2
   double I_slug_scalar = 0.4 * this->massInit * std::pow(this->currentSlugRadius, 2);
   Eigen::Matrix3d I_slug_matrix = I_slug_scalar * Eigen::Matrix3d::Identity();
   
   Eigen::Matrix3d rTilde = eigenTilde(this->r_SB_B);
   this->effProps.IEffPntB_B = I_slug_matrix - this->massInit * rTilde * rTilde;
   
   // Inertia Derivative
   double r_dot_scalar = (r_norm > 1e-6) ? r_rel.dot(v_rel) / r_norm : 0.0;
   double L_dot = -r_dot_scalar;
   
   double I_slug_dot_scalar = 0.8 * this->massInit * this->currentSlugRadius * L_dot;
   Eigen::Matrix3d I_slug_dot_matrix = I_slug_dot_scalar * Eigen::Matrix3d::Identity();
   
   Eigen::Matrix3d vTilde = eigenTilde(this->v_SB_B);
   this->effProps.IEffPrimePntB_B = I_slug_dot_matrix - this->massInit * (vTilde * rTilde + rTilde * vTilde);
}

void MovingPulsatingBall::updateEnergyMomContributions(double integTime, Eigen::Vector3d & rotAngMomPntCContr_B, double & rotEnergyContr, Eigen::Vector3d omega_BN_B) {
   this->omega_BN_B = omega_BN_B; 
   
   Eigen::Vector3d omega_s = this->omegaState->getState();
   double I_slug_scalar = 0.4 * this->massInit * std::pow(this->currentSlugRadius, 2);
   Eigen::Matrix3d I_slug = I_slug_scalar * Eigen::Matrix3d::Identity();
   
   Eigen::Vector3d H_internal = I_slug * omega_s + this->massInit * this->r_SB_B.cross(this->v_SB_B);
   
   rotAngMomPntCContr_B = this->effProps.IEffPntB_B * omega_BN_B + H_internal;
   rotEnergyContr = 0.0; 
}

void MovingPulsatingBall::computeDerivatives(double integTime, Eigen::Vector3d rDDot_BN_N, Eigen::Vector3d omegaDot_BN_B, Eigen::Vector3d sigma_BN) {
   // --- 1. Retrieve States ---
   Eigen::Vector3d r_vec = this->posState->getState();       
   Eigen::Vector3d v_vec = this->velState->getState();       
   Eigen::Vector3d omega_s = this->omegaState->getState();  
   Eigen::Vector3d omega_hub = this->omega_BN_B;            
   Eigen::Vector3d omegaDot_hub = omegaDot_BN_B;            
   
   double r_norm = r_vec.norm();
   // [Fix 2] Robust Singularity Protection
   if (r_norm < 1e-3) { 
       r_norm = 1e-3; 
       // If vector is too small, pick arbitrary direction to avoid NaN from normalized()
       if (r_vec.norm() < 1e-6) {
           r_vec = Eigen::Vector3d(0.0, 0.0, 1.0) * r_norm;
       } else {
           r_vec = r_vec.normalized() * r_norm; 
       }
   }
   
   double L = this->radiusTank - r_norm; 
   Eigen::Vector3d e_i = r_vec.normalized(); 
   double r_dot_scalar = r_vec.dot(v_vec) / r_norm; 
   
   // --- 2. Calculate Kinematics ---
   Eigen::Vector3d w_i = e_i.cross(v_vec) / r_norm;
   Eigen::Vector3d omega_ri = e_i.dot(omega_s) * e_i;
   
   // --- 3. Calculate Forces (Zhang Eq. 6 + Soft Barrier) ---
   
   // Standard MPBM Terms
   Eigen::Vector3d term1_vec = e_i.cross(omega_hub + w_i);
   double term1 = r_norm * term1_vec.squaredNorm(); 
   double term2 = (omega_hub.cross(this->r_TB_B)).dot(omega_hub.cross(e_i));
   double term3 = -(this->r_TB_B.cross(e_i)).dot(omegaDot_hub);
   Eigen::Vector3d total_omega = omega_s + omega_hub;
   double term4 = (this->massInit / 4.0) * L * total_omega.squaredNorm();
   double term5 = -5.0 * M_PI * this->surfaceTension * L;
   
   double N_val = (3.0 * this->massInit / 8.0) * (term1 + term2 + term3) + term4 + term5;
   
   // [SOFT PENALTY LOGIC]
   // [Fix 5] Ensure L is within physical bounds for penetration calculation
   if (L <= this->radiusSlugMin) {
      double penetration = this->radiusSlugMin - L; // Positive when penetrating
      
      // Cap penetration to avoid extreme forces if numerical integration overshoots significantly
      if (penetration > 2.0 * this->radiusSlugMin) penetration = 2.0 * this->radiusSlugMin;

      double speed_into_wall = -r_dot_scalar;       // Positive when moving towards L=0
      
      double spring_force = this->k_barrier * penetration;
      double damping_force = this->c_barrier * speed_into_wall;
      
      if (speed_into_wall < 0) damping_force = 0; 

      N_val += (spring_force + damping_force);
   }
   
   // Friction Force F_bi (Zhang Eq. 7)
   // [Fix 3] Clamping to prevent division by zero / massive forces
   double epsilon = 0.05; // 5cm floor
   double L_clamped = (L < epsilon) ? epsilon : L;
   double coef_F = (6500.0 * this->kinematicViscosity * this->massInit) / (L_clamped * L_clamped);
   
   Eigen::Vector3d inner_vec = e_i.cross(v_vec) + L * omega_s;
   Eigen::Vector3d F_bi = -coef_F * e_i.cross(inner_vec);
   
   Eigen::Vector3d F_Li = N_val * e_i + F_bi;

   // Interaction Torque T_Li (Zhang Eq. 8)
   double omega_s_norm = omega_s.norm();
   if(omega_s_norm < 1e-8) omega_s_norm = 1e-8;

   double f_ac = 0.36 * std::pow(this->massInit, 4.0/3.0) 
                * std::pow(this->rho, 1.0/6.0) 
                * std::sqrt(this->kinematicViscosity)
                * std::pow(L / this->radiusSlugMin, 2.0) 
                * std::sqrt(omega_s_norm);
                
   Eigen::Vector3d T_Li = f_ac * (this->t_sr * omega_ri + (1.0 - this->t_sr)*(omega_s - omega_ri));
   
   // --- 4. Solve Dynamics ---
   
   // A. Translational
   Eigen::MRPd sigmaMRP(sigma_BN);
   Eigen::Matrix3d dcm_BN = sigmaMRP.toRotationMatrix().transpose();
   Eigen::Vector3d a_hub_B = dcm_BN * rDDot_BN_N;
   Eigen::Vector3d r_Si = this->r_TB_B + r_vec; 
   
   Eigen::Vector3d inertial_acc = a_hub_B 
                              + omegaDot_hub.cross(r_Si) 
                              + omega_hub.cross(omega_hub.cross(r_Si)) 
                              + 2.0 * omega_hub.cross(v_vec);
                              
   Eigen::Vector3d v_dot = (-F_Li / this->massInit) - inertial_acc;
   
   // B. Rotational
   // [Fix 4] Inertia Singularity Protection
   double L_inertia = (L < 1e-3) ? 1e-3 : L;
   double I_s = 0.4 * this->massInit * L_inertia * L_inertia;
   
   Eigen::Vector3d RHS_Eq4 = -T_Li - L * e_i.cross(F_Li);
   Eigen::Vector3d LHS_Coriolis = (0.4 * this->massInit * L) * (-2.0 * r_dot_scalar) * (omega_s + omega_hub);
   Eigen::Vector3d LHS_Hub = I_s * (omegaDot_hub + omega_hub.cross(omega_s));
   
   Eigen::Vector3d Torque_Net = RHS_Eq4 - LHS_Coriolis - LHS_Hub;
   Eigen::Vector3d omega_s_dot = Torque_Net / I_s;
   
   // --- 5. Save Derivatives ---
   this->posState->setDerivative(v_vec);      
   this->velState->setDerivative(v_dot);      
   this->omegaState->setDerivative(omega_s_dot); 

   // [Validation] Store Interaction Torque for Logging
   this->current_T_Li = T_Li;
}

void MovingPulsatingBall::UpdateState(uint64_t CurrentSimNanos) {
    // [Validation] Log internal states to output message
    SCStatesMsgPayload outMsgBuffer;
    memset(&outMsgBuffer, 0x0, sizeof(SCStatesMsgPayload)); 

    Eigen::Vector3d r_vec = this->posState->getState();
    Eigen::Vector3d v_vec = this->velState->getState();
    Eigen::Vector3d t_vec = this->current_T_Li;

    // Field Mapping:
    // r_BN_N -> Slug Position
    eigenVector3d2CArray(r_vec, outMsgBuffer.r_BN_N);
    
    // v_BN_N -> Slug Velocity
    eigenVector3d2CArray(v_vec, outMsgBuffer.v_BN_N);
    
    // omega_BN_B -> Interaction Torque (Important for Validation)
    eigenVector3d2CArray(t_vec, outMsgBuffer.omega_BN_B); 

    // totalMass -> Slug Mass
    // outMsgBuffer.totalMass = this->massInit;

    this->mpbmOutMsg.write(&outMsgBuffer, this->moduleID, CurrentSimNanos);
}