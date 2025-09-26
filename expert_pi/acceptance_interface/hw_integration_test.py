from expert_pi import grpc_client


def grpc_test():

    probe_current = grpc_client.illumination.get_current()*1e12
    convergence_angle = grpc_client.illumination.get_convergence_half_angle()*1e3
    stage_xy = grpc_client.stage.get_x_y()

    results = {"Probe current pA":probe_current,"Convergence angle (mrad)":convergence_angle,"Stage position":stage_xy}

    return results