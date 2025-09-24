import torch
import numpy as np
from scipy import interpolate
from collections import OrderedDict
import argparse

def interpolate_relative_pos_embed(rel_pos_bias, dst_num_pos, param_name=''):
    """
    Interpolates the relative position bias table to a new size.
    Source: https://github.com/microsoft/unilm/blob/8a0a1c1f4e7326938ea7580a00d56d7f17d65612/beit/run_class_finetuning.py#L348
    """
    src_num_pos, num_attn_heads = rel_pos_bias.size()
    src_size = int(src_num_pos ** 0.5)
    dst_size = int(dst_num_pos ** 0.5)

    if src_size != dst_size:
        print(f"Interpolating position embedding {param_name} from {src_size}x{src_size} to {dst_size}x{dst_size}")

        def geometric_progression(a, r, n):
            return a * (1.0 - r ** n) / (1.0 - r)

        left, right = 1.01, 1.5
        while right - left > 1e-6:
            q = (left + right) / 2.0
            gp = geometric_progression(1, q, src_size // 2)
            if gp > dst_size // 2:
                right = q
            else:
                left = q

        dis = []
        cur = 1
        for i in range(src_size // 2):
            dis.append(cur)
            cur += q ** (i + 1)

        r_ids = [-_ for _ in reversed(dis)]
        x = r_ids + [0] + dis
        y = r_ids + [0] + dis

        t = dst_size // 2.0
        dx = np.arange(-t, t + 0.1, 1.0)
        dy = np.arange(-t, t + 0.1, 1.0)

        all_rel_pos_bias = []
        for i in range(num_attn_heads):
            z = rel_pos_bias[:, i].view(src_size, src_size).float().numpy()
            f = interpolate.interp2d(x, y, z, kind='cubic')
            all_rel_pos_bias.append(
                torch.Tensor(f(dx, dy)).contiguous().view(-1, 1).to(rel_pos_bias.device)
            )

        rel_pos_bias = torch.cat(all_rel_pos_bias, dim=-1)

    return rel_pos_bias

def process_raw_xvlm_state_dict(state_dict, window_size):
    """
    Processes the raw XVLM state dictionary by renaming keys, removing unnecessary
    ones, and interpolating position embeddings.
    """
    new_state_dict = OrderedDict()

    for key, value in state_dict.items():
        if 'relative_position_bias_table' in key:
            dst_num_pos = (2 * window_size - 1) ** 2
            new_state_dict[key] = interpolate_relative_pos_embed(value, dst_num_pos, param_name=key)
        elif 'relative_position_index' in key or 'attn_mask' in key:
            continue
        elif 'text_encoder.bert.' in key:
            new_key = key.replace('bert.', '')
            new_state_dict[new_key] = value
        elif 'bbox_head.' in key:
            new_key = key.replace('bbox_head.', 'bbox_head.fc.')
            new_state_dict[new_key] = value
        elif 'itm_head.' in key:
            new_key = key.replace('itm_head.', 'itm_head.fc.')
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value

    return new_state_dict

def print_state_dict_info(state_dict):
    """
    Prints the keys and shapes of a state dictionary in a formatted table.
    """
    max_key_len = max(len(str(key)) for key in state_dict.keys())
    max_shape_len = max(len(str(v.shape)) for v in state_dict.values() if hasattr(v, 'shape'))

    header_format = f"\n{{:<{max_key_len}}}   {{:<{max_shape_len}}}"
    print(header_format.format("Key", "Shape"))
    print("-" * (max_key_len + max_shape_len + 3))

    for key, value in state_dict.items():
        shape_str = str(getattr(value, 'shape', 'N/A'))
        row_format = f"{{:<{max_key_len}}}   {{:<{max_shape_len}}}"
        print(row_format.format(key, shape_str))

def main(args):
    """
    Main function to load, process, and save the model state dictionary.
    """
    xvlm_state_dict = torch.load(args.input_path, map_location="cpu")
    processed_state_dict = process_raw_xvlm_state_dict(xvlm_state_dict['model'], args.window_size)
    
    torch.save(processed_state_dict, args.output_path)

    print("\nProcessed State Dictionary Contents:")
    print_state_dict_info(processed_state_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Process XVLM model state dictionary for use in MMCV frameworks."
    )

    parser.add_argument(
        '--input_path',
        type=str,
        default="pretrain/16m_base_model_state_step_199999.th",
        help="Path to the input XVLM model checkpoint (.th file)."
    )
    
    parser.add_argument(
        '--output_path',
        type=str,
        default="pretrain/16m_base_model_state_step_199999_(xvlm2mmcv).pth",
        help="Path to save the processed model checkpoint (.pth file)."
    )

    parser.add_argument(
        '--window_size',
        type=int,
        default=12,
        help="Window size for position embedding interpolation."
    )

    parser.add_argument(
        '--image_res',
        type=int,
        default=384,
        help="Image resolution. (Currently unused in script logic but kept for context)."
    )

    parser.add_argument(
        '--patch_size',
        type=int,
        default=32,
        help="Patch size. (Currently unused in script logic but kept for context)."
    )

    args = parser.parse_args()
    main(args)