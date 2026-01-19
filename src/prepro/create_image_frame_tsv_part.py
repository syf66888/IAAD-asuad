import os.path as op
import json
import os
import sys
from pathlib import Path    
import argparse
from tqdm import tqdm
import numpy as np
import multiprocessing as mp
pythonpath = os.path.abspath(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
print(pythonpath)
sys.path.insert(0, pythonpath)
from src.utils.tsv_file_ops import tsv_writer
from PIL import Image
import io
import base64
import cv2


def resize_and_to_binary(img_path, target_image_size):
    if img_path is None:
        if target_image_size < 0:
            target_image_size = 256
        resized = np.zeros((target_image_size, target_image_size, 3), dtype = "uint8")
        s = (target_image_size, target_image_size)
    else:
        # im = Image.open(img_path)
        im = cv2.imread(img_path)
        height, width, channels = im.shape
        s = (width, height)
        if target_image_size > 0:
            s = min(width, height)
                
            r = target_image_size / s
            s = (round(r * width), round(r * height))
            # im = im.resize(s)
            resized = cv2.resize(im, s)
        else:
            resized = im

    # binary = io.BytesIO()
    # im.save(binary, format='JPEG')
    # binary = binary.getvalue()
    binary = cv2.imencode('.jpg', resized)[1].tobytes()
    encoded_base64 = base64.b64encode(binary)
    return encoded_base64, s


def load_tsv_to_mem(tsv_file, sep='\t'):
    data = []
    with open(tsv_file, 'r') as fp:
        for _, line in enumerate(fp):
            data.append([x.strip() for x in line.split(sep)])
    return data


def get_image_binaries(list_of_paths, image_size=56):
    batch = []
    is_None = [v is None for v in list_of_paths]
    assert not any(is_None) or all(is_None)
    for img_path in list_of_paths:
        if img_path is None or isinstance(img_path, str):
            x, shape = resize_and_to_binary(img_path, target_image_size=image_size)
        else:
            raise ValueError(f'img_path not str, but {type(img_path)}')
        batch.append(x)
    return batch, shape


def prepare_single_video_frames(caption_id, vid_path, num_frames=32):
    previous_image_path = None
    images = []
    local_data_path = vid_path.replace("datasets", "_datasets")
    if not op.exists(local_data_path) and not op.exists(vid_path):
        # print(f'{vid_path} does not exists')
        images = [None]*num_frames
        return None

    video_id = Path(vid_path).stem
    for i in range(num_frames):
        current_image_path = op.join(data_path, f'{video_id}_frame{(i+1):04d}.jpg')
        if not op.exists(current_image_path):
            print(f'{current_image_path} does not exists')
            exit()
            if previous_image_path:
                current_image_path = previous_image_path 
            else:
                print(f'The first image for {video_id} does not exists')
                images = [None]*num_frames
                return images
        images.append(current_image_path)
        previous_image_path = current_image_path
    return images


def process_video_chunk(item, image_size=56, num_frames=32):
    # line_items = []
    # for item in items:
    caption_id = item['id']
    vid_path = 'datasets/own/raw_videos/clip2caption/' + item['vidName'] + '.mp4'
    
    
    images = prepare_single_video_frames(caption_id, vid_path, num_frames)
    if images == None:
        return None
    image_binaries, image_shape = get_image_binaries(images, image_size)
    
    resolved_data_vid_id = str(caption_id).zfill(5)+'/'+ item['vidName'] 
    line_item = [str(resolved_data_vid_id), json.dumps({"class": -1, "width": image_shape[0], "height": image_shape[1]})]
    line_item += image_binaries
    return line_item
    #     line_items.append(line_item)
    # return line_items


def main():
    output_folder = f"datasets/own/frame_tsv"
    os.makedirs(output_folder, exist_ok=True)
    # To generate a tsv file:
    # data_path: path to raw video files
    global data_path

    image_size = 256

    num_frames = 32
    data_path =f"datasets/own/32frames/" 

    num_workers = 32
    video_info_tsv = 'datasets/own/captions_ccd.json'
    
    split = 'testing'
    
    if split == 'training':
        data = json.load(open(video_info_tsv))['annotations'][:700]
    elif split == 'validation':
        data = json.load(open(video_info_tsv))['annotations'][700:800]
    elif split == 'testing':
        data = json.load(open(video_info_tsv))['annotations'][800:]
    else:
        exit("split error")
    # data = load_tsv_to_mem(f'datasets/{args.dataset}/{args.split}.img.tsv')

    if num_workers > 0 :
        resolved_visual_file = f"{output_folder}/{split}_{num_frames}frames_img_size{image_size}.img.tsv"
        print("generating visual file for", resolved_visual_file)

        from functools import partial
        worker = partial(
            process_video_chunk, image_size=image_size, num_frames=num_frames)

        def gen_rows():
            with mp.Pool(num_workers) as pool, tqdm(total=len(data)) as pbar:
                for _, line_item in enumerate(
                        pool.imap(worker, data, chunksize=8)):
                    pbar.update(1)
                    if line_item is not None:
                        yield(line_item)

        tsv_writer(gen_rows(), resolved_visual_file)
    else:
        for idx, d in tqdm(enumerate(data),
                           total=len(data), desc="extracting frames from video"):
            process_video_chunk(d, image_size=image_size, num_frames=num_frames)



if __name__ == '__main__':
    main()


