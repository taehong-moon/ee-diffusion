import torch


def default_scheduler(x, timesteps, c, blocks):
    for block in blocks:
        x = block(x, c)      # (N, T, D)
    return x


def build_forward_scheduler(schedule, is_sampling):

    if schedule == "demo":
        exit_nums = [28, 28, 24, 24, 20, 20, 16, 16, 12, 12]

    exit_num_1, exit_num_2, exit_num_3, exit_num_4, exit_num_5, exit_num_6, exit_num_7, exit_num_8, exit_num_9, exit_num_10 = exit_nums

    def get_early_exit_idx(time_idx):
        if (time_idx >= 0) and (time_idx < 100):
            return exit_num_1
        elif (time_idx >= 100) and (time_idx < 200):
            return exit_num_2
        elif (time_idx >= 200) and (time_idx < 300):
            return exit_num_3
        elif (time_idx >= 300) and (time_idx < 400):
            return exit_num_4
        elif (time_idx >= 400) and (time_idx < 500):
            return exit_num_5
        elif (time_idx >= 500) and (time_idx < 600):
            return exit_num_6
        elif (time_idx >= 600) and (time_idx < 700):
            return exit_num_7
        elif (time_idx >= 700) and (time_idx < 800):
            return exit_num_8
        elif (time_idx >= 800) and (time_idx < 900):
            return exit_num_9
        else:
            return exit_num_10
    
    def drop_scheduler(x, timesteps, c, blocks):
        if not is_sampling:
            gpu_t_size = int(timesteps.size(0))
            cond_t = [timesteps[idx].item() for idx in range(gpu_t_size)]
            
            early_exit_x = {}
            for idx, block in enumerate(blocks):
                x = block(x, c)      # (N, T, D)
                early_exit_x[idx+1] = x # TODO Debug This Part!
            
            selected_x = []
            for idx, t in enumerate(cond_t):
                exit_num = get_early_exit_idx(t)
                fix_t_h = early_exit_x[exit_num][idx, :, :].unsqueeze(0) # (1, T, D)
                selected_x.append(fix_t_h)
            
            x = torch.cat(selected_x, dim=0)
        else:
            t = timesteps[0].item()
            exit_num = get_early_exit_idx(t)
            for block in blocks[:exit_num]:
                x = block(x, c)      # (N, T, D)
        
        return x
    
    return drop_scheduler