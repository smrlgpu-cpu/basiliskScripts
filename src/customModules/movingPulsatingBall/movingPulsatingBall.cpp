/*
* MPBM State Effector: Fluid-Like Physics (Viscous Wall & Soft Saturation)
* * Concept:
* 1. Liquid doesn't bounce -> k_barrier approx 0, High c_barrier.
* 2. Baffled Tank Effect -> Use tanh() to smoothly limit forces (Soft Yielding).
* 3. RL Friendly -> Continuous gradients everywhere (No hard cuts).
*/

#include "movingPulsatingBall.h"
#include "architecture/utilities/linearAlgebra.h"
#include "architecture/utilities/avsEigenSupport.h"
#include <iostream>
#include <cmath>
#include <cstring>
#include <algorithm>

// Soft Saturation Function (Gradient Friendly)
// Converts infinite spike forces into a smooth curve approaching the limit.
// y = limit * tanh(x / limit)
Eigen::Vector3d softSaturate(Eigen::Vector3d vec, double limit) {
   double norm = vec.norm();
   if (norm < 1e-9) return vec;
   
   // Smooth compression of magnitude
   double saturated_norm = limit * std::tanh(norm / limit);
   
   return vec * (saturated_norm / norm);
}

Eigen::Matrix3d eigenTilde(Eigen::Vector3d v) {
   Eigen::Matrix3d m;
   m << 0, -v[2], v[1],
      v[2], 0, -v[0],
      -v[1], v[0], 0;
   return m;
}

MovingPulsatingBall::MovingPulsatingBall() {
   this->massInit = 100.0;
   this->radiusTank = 0.50;
   this->radiusSlugMin = 0.10;
   this->kinematicViscosity = 2.839e-6;
   this->surfaceTension = 0.066;
   this->rho = 1004.0;
   
   // Circulation factor
   this->t_sr = 0.1; 
   
   // Unused member vars
   this->k_barrier = 0.0; 
   this->c_barrier = 0.0;

   this->r_TB_B.setZero();
   this->r_Init_B << 0.0, 0.0, 0.1;
   this->v_Init_B.setZero();
   
   this->nameOfPosState = "mpbmPos";
   this->nameOfVelState = "mpbmVel";
   this->nameOfOmegaState = "mpbmOmega";
   
   this->current_T_Li.setZero();
}

MovingPulsatingBall::~MovingPulsatingBall() {}

void MovingPulsatingBall::Reset(uint64_t CurrentSimNanos) {
   this->effProps.mEff = this->massInit;
   this->current_T_Li.setZero();
   
   MPBMStateMsgPayload outMsgBuffer = {}; 
   eigenVector3d2CArray(this->r_Init_B, outMsgBuffer.r_Slug_B);
   eigenVector3d2CArray(this->v_Init_B, outMsgBuffer.v_Slug_B);
   eigenVector3d2CArray(this->current_T_Li, outMsgBuffer.T_Interaction);
   outMsgBuffer.mass = this->massInit;
   this->mpbmOutMsg.write(&outMsgBuffer, this->moduleID, CurrentSimNanos);
}

void MovingPulsatingBall::registerStates(DynParamManager& states) {
   Eigen::Vector3d initOmega; initOmega.setZero();
   this->posState = states.registerState(3, 1, this->nameOfPosState);
   this->posState->setState(this->r_Init_B);
   this->velState = states.registerState(3, 1, this->nameOfVelState);
   this->velState->setState(this->v_Init_B);
   this->omegaState = states.registerState(3, 1, this->nameOfOmegaState);
   this->omegaState->setState(initOmega);
}

void MovingPulsatingBall::linkInStates(DynParamManager& states) {}

void MovingPulsatingBall::updateEffectorMassProps(double integTime) {
   Eigen::Vector3d r_rel = this->posState->getState();
   Eigen::Vector3d v_rel = this->velState->getState();
   
   double r_norm = r_rel.norm();
   if(r_norm > this->radiusTank) r_norm = this->radiusTank - 1e-4;
   
   this->currentSlugRadius = this->radiusTank - r_norm;
   this->r_SB_B = this->r_TB_B + r_rel;
   this->v_SB_B = v_rel;
   
   this->effProps.mEff = this->massInit;
   this->effProps.rEff_CB_B = this->r_SB_B;
   this->effProps.rEffPrime_CB_B = this->v_SB_B;
   
   double I_slug_scalar = 0.4 * this->massInit * std::pow(this->currentSlugRadius, 2);
   Eigen::Matrix3d I_slug_matrix = I_slug_scalar * Eigen::Matrix3d::Identity();
   Eigen::Matrix3d rTilde = eigenTilde(this->r_SB_B);
   this->effProps.IEffPntB_B = I_slug_matrix - this->massInit * rTilde * rTilde;
   
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
   // 1. Retrieve States
   Eigen::Vector3d r_vec = this->posState->getState();
   Eigen::Vector3d v_vec = this->velState->getState();
   Eigen::Vector3d omega_s = this->omegaState->getState();
   Eigen::Vector3d omega_hub = this->omega_BN_B;
   Eigen::Vector3d omegaDot_hub = omegaDot_BN_B;
   
   double r_norm = r_vec.norm();
   if (r_norm < 1e-4) {
      r_vec = (r_norm < 1e-6) ? Eigen::Vector3d(0,0,1) * 1e-4 : r_vec.normalized() * 1e-4;
      r_norm = 1e-4;
   }
   
   double L = this->radiusTank - r_norm;
   Eigen::Vector3d e_i = r_vec.normalized();
   double r_dot_scalar = r_vec.dot(v_vec) / r_norm;
   double L_calc = (L < 0.01) ? 0.01 : L;

   // 2. Kinematics
   Eigen::Vector3d w_i = e_i.cross(v_vec) / r_norm;
   Eigen::Vector3d omega_ri = e_i.dot(omega_s) * e_i;

   // 3. Raw Forces (Zhang Eq. 6) - Standard Fluid Model
   Eigen::Vector3d term1_vec = e_i.cross(omega_hub + w_i);
   double term1 = r_norm * term1_vec.squaredNorm();
   double term2 = (omega_hub.cross(this->r_TB_B)).dot(omega_hub.cross(e_i));
   double term3 = -(this->r_TB_B.cross(e_i)).dot(omegaDot_hub);
   Eigen::Vector3d total_omega = omega_s + omega_hub;
   double term4 = (this->massInit / 4.0) * L * total_omega.squaredNorm();
   double term5 = -5.0 * M_PI * this->surfaceTension * L;
   
   double N_val_raw = (3.0 * this->massInit / 8.0) * (term1 + term2 + term3) + term4 + term5;

   // =========================================================================
   // Viscous Wall Model (No Bounce, Just Stick)
   // Liquid dissipates energy -> "Muddy" wall interaction.
   // =========================================================================
   
   double barrier_force = 0.0;
   
   // Smooth activation starting 10cm from wall
   if (L < 0.02) { 
      double dist = (L < 0.001) ? 0.001 : L;
      double speed_out = r_dot_scalar; // +: towards wall
      
      // 1. Viscous Damping (Dominant Term)
      // Damping increases as 1/dist -> Mimics fluid squeeze film effect
      // Viscosity becomes infinite as gap closes
      if (speed_out > 0) {
         double c_mud = 100.0 / dist; // Hyperbolic damping
         
         // Limit Damping Force
         if (c_mud > 1e6) c_mud = 1e6;

         barrier_force += c_mud * speed_out;
      }
      
      // 2. Weak Spring (Just to define volume)
      // Very weak stiffness just to guide it back eventually
      double penetration = 0.10 - L;
      barrier_force += 100.0 * penetration; 
      
      N_val_raw -= barrier_force;
   }
   
   // Friction Force
   double coef_F = (6500.0 * this->kinematicViscosity * this->massInit) / (L_calc * L_calc);
   Eigen::Vector3d inner_vec = e_i.cross(v_vec) + L_calc * omega_s;
   Eigen::Vector3d F_bi_raw = -coef_F * e_i.cross(inner_vec);
   
   // Interaction Torque
   double omega_s_norm = omega_s.norm();
   if(omega_s_norm < 1e-8) omega_s_norm = 1e-8;
   double f_ac = 0.36 * std::pow(this->massInit, 4.0/3.0) 
               * std::pow(this->rho, 1.0/6.0) 
               * std::sqrt(this->kinematicViscosity)
               * std::pow(L_calc / this->radiusSlugMin, 2.0) 
               * std::sqrt(omega_s_norm);
   Eigen::Vector3d T_Li_raw = f_ac * (this->t_sr * omega_ri + (1.0 - this->t_sr)*(omega_s - omega_ri));

   // =========================================================================
   // Soft Saturation (Tanh)
   // Simulates "Yielding" of fluid/baffles. 
   // Smooth gradients for RL, realistic limits for RW.
   // =========================================================================

   // 1. Soft Saturate Torque -> Limit to ~2.0 Nm
   // "Baffled tank limit": fluid cannot exert more torque than this before breaking into turbulence.
   Eigen::Vector3d T_Li_Physical = softSaturate(T_Li_raw, 2.0);

   // 2. Soft Saturate Friction -> Limit to ~20 N
   // Shear stress limit of fluid against wall.
   Eigen::Vector3d F_bi_Physical = softSaturate(F_bi_raw, 20.0);

   // 3. Soft Saturate Normal Force -> Limit to ~50 N
   // "Splash" limit. Water creates a splash, not a hard impact.
   // We apply tanh to the SCALAR value of N_val
   double N_val_Physical = 50.0 * std::tanh(N_val_raw / 50.0);

   // Combined Physical Force
   Eigen::Vector3d F_Li_Physical = N_val_Physical * e_i + F_bi_Physical;

   // --- 4. Solve Dynamics (Conserved Physics) ---
   
   // A. Translational
   Eigen::MRPd sigmaMRP(sigma_BN);
   Eigen::Matrix3d dcm_BN = sigmaMRP.toRotationMatrix().transpose();
   Eigen::Vector3d a_hub_B = dcm_BN * rDDot_BN_N;
   Eigen::Vector3d r_Si = this->r_TB_B + r_vec; 
   Eigen::Vector3d inertial_acc = a_hub_B + omegaDot_hub.cross(r_Si) + omega_hub.cross(omega_hub.cross(r_Si)) + 2.0 * omega_hub.cross(v_vec);
   
   // Both Slug and Body feel the "Soft Saturated" forces
   Eigen::Vector3d v_dot = (-F_Li_Physical / this->massInit) - inertial_acc;
   
   // Tunneling Guard (Emergency Reset only)
   if (L < -0.001) {
      double v_radial = v_vec.dot(e_i);
      if (v_radial > 0) v_dot -= 2000.0 * v_radial * e_i;
   }
   
   // Global Damping (Fluid Viscosity)
   v_dot -= 0.1 * v_vec;

   // B. Rotational
   double I_s = 0.4 * this->massInit * L_calc * L_calc;

   // Prevent Division by Zero
   if (I_s < 1e-6) I_s = 1e-6;

    Eigen::Vector3d RHS_Eq4 = -T_Li_Physical - L_calc * e_i.cross(F_Li_Physical);
   Eigen::Vector3d LHS_Coriolis = (0.4 * this->massInit * L_calc) * (-2.0 * r_dot_scalar) * (omega_s + omega_hub);
   Eigen::Vector3d LHS_Hub = I_s * (omegaDot_hub + omega_hub.cross(omega_s));
   
   Eigen::Vector3d Torque_Net = RHS_Eq4 - LHS_Coriolis - LHS_Hub;
   Eigen::Vector3d omega_s_dot = Torque_Net / I_s;

   // Clamp Angular Acceleration
   double omega_s_dot_norm = omega_s_dot.norm();
   if (omega_s_dot_norm > 100.0) {
      omega_s_dot = omega_s_dot * (100.0 / omega_s_dot_norm);
   }

   // Rotational Damping
   omega_s_dot -= 0.5 * omega_s; 

   // 5. Save Derivatives
   this->posState->setDerivative(v_vec);
   this->velState->setDerivative(v_dot);
   this->omegaState->setDerivative(omega_s_dot);
   
   // Log the Physical (Soft Saturated) Torque
   this->current_T_Li = T_Li_Physical;
}

void MovingPulsatingBall::UpdateState(uint64_t CurrentSimNanos) {
   MPBMStateMsgPayload outMsgBuffer = {}; 
   Eigen::Vector3d r_vec = this->posState->getState();
   Eigen::Vector3d v_vec = this->velState->getState();
   
   // Use member variable that stores the latest computed torque
   Eigen::Vector3d t_vec = this->current_T_Li;

   eigenVector3d2CArray(r_vec, outMsgBuffer.r_Slug_B);
   eigenVector3d2CArray(v_vec, outMsgBuffer.v_Slug_B);
   eigenVector3d2CArray(t_vec, outMsgBuffer.T_Interaction);
   outMsgBuffer.mass = this->massInit;
   
   this->mpbmOutMsg.write(&outMsgBuffer, this->moduleID, CurrentSimNanos);
}