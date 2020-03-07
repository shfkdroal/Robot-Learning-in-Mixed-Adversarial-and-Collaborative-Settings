from mujoco_py import load_model_from_path, MjSim, MjViewer
import mujoco_py.cymj as cymj
import copy
import os
import numpy as np
import imageio

def inverse_kinematics(model_name,
                        body_name, target_pos,target_quat,
                        max_steps=100,
                        rot_weight=1.0,
                        tol=1e-14, 
                        regularization_threshold=0.1,
                        regularization_strength=3e-2,
                        max_update_norm=2.0,
                        progress_thresh=20.0,
                        image_name='test'):
    model = load_model_from_path(model_name)
    sim=MjSim(model)
    # viewer=MjViewer(sim)
    sim.step()
    for step in range(max_steps):
        body_xmat=sim.data.get_body_xmat(body_name)
        body_xpos=sim.data.get_body_xpos(body_name)
        err_norm=0.0
        err_pos=target_pos-body_xpos
        err_norm += np.linalg.norm(err_pos)

        quat=np.empty(4)
        neg_quat=np.empty(4)
        err_rot_quat=np.empty(4)
        err_rot=np.empty(3)

        xmat=np.reshape(body_xmat,[9,])
        cymj._mju_mat2Quat(quat,xmat)
        cymj._mju_negQuat(neg_quat,quat)
        cymj._mju_mulQuat(err_rot_quat, target_quat, neg_quat)
        cymj._mju_quat2Vel(err_rot, err_rot_quat, 1)
        #######!!!!test!!!!
        # err_rot=np.zeros(err_rot.shape)
        #######
        err_norm += np.linalg.norm(err_rot) * rot_weight
        err=np.concatenate((err_pos,err_rot))

        if err_norm < tol:
            print('Converged after %i steps: err_norm=%3g' % (step, err_norm))
            success = True
            break
        else:
            jacp = np.zeros(3 * sim.model.nv)
            sim.data.get_body_jacp(body_name, jacp=jacp)
            jacr = np.zeros(3 * sim.model.nv)
            sim.data.get_body_jacr(body_name, jacr=jacr)
            jacp=np.reshape(jacp,(3,sim.model.nv))
            jacr=np.reshape(jacr,(3,sim.model.nv))
            jac_joints=np.concatenate((jacp,jacr),axis=0)
            reg_strength = (regularization_strength if err_norm > regularization_threshold else 0.0)

            update_joints = nullspace_method(jac_joints, err, reg_strength=reg_strength)

            update_norm = np.linalg.norm(update_joints)
            # Check whether we are still making enough progress, and halt if not.
            progress_criterion = err_norm / update_norm
            if progress_criterion > progress_thresh:
                print('Step %2i: err_norm / update_norm (%3g) > '
                    'tolerance (%3g). Halting due to insufficient progress'
                    % (step, progress_criterion, progress_thresh))
                break
            if update_norm > max_update_norm:
                update_joints *= max_update_norm / update_norm
        cymj._mj_integratePos(sim.model, sim.data.qpos, update_joints, 1)
        cymj._mj_fwdPosition(sim.model, sim.data)
    qpos_out=copy.deepcopy(sim.data.qpos[:])
    return qpos_out
        
def nullspace_method(jac_joints,delta,reg_strength):
    hess_approx = jac_joints.T.dot(jac_joints)
    joint_delta = jac_joints.T.dot(delta)
    if reg_strength > 0:
    # L2 regularization
        hess_approx += np.eye(hess_approx.shape[0]) * reg_strength
        return np.linalg.solve(hess_approx, joint_delta)
    else:
        return np.linalg.lstsq(hess_approx, joint_delta,rcond=-1)[0]