import os, glob, argparse, pickle
from tqdm import tqdm

lang2code = {'english': 'en', 'spanish': 'es', 'chinese': 'zh-CN', 'italian': 'it', 'polish': 'pl'}

def pkl2dict(pkl_path):
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)

def main(args):
    # pkl_path   = os.path.join(args.root_path, f"{lang2code[args.language]}_list.pkl")
    # file_dict  = pkl2dict(pkl_path)
    save_path  = os.path.join(args.root_path, args.language, lang2code[args.language], 'clips_wav')
    os.makedirs(save_path, exist_ok=True)
    audio_path = os.path.join(args.root_path, args.language, lang2code[args.language], 'clips')
    assert os.path.exists(audio_path)
    
    # for keyword, files in file_dict.items():
    #     os.makedirs(os.path.join(save_path, keyword), exist_ok=True)
    #     for file in files:
    #         if file.split('.')[-1] == 'opus':
    #             opus_path = os.path.join(audio_path, file)
    #             wav_path = opus_path.replace('.opus', '.wav').replace('clips', 'clips_wav')
    #             os.system(f'ffmpeg -i "{opus_path}" -ar {args.sample_rate} -vn "{wav_path}"')
    opus_paths = glob.glob(audio_path + '/*/*.opus', recursive=True)
    wav_paths = [opus.replace('.opus', '.wav').replace('clips', 'clips_wav') for opus in opus_paths]

    for opus_path, wav_path in tqdm(zip(opus_paths, wav_paths)):
        os.makedirs(('/').join(wav_path.split('/')[:-1]), exist_ok=True)
        os.system(f'ffmpeg -i "{opus_path}" -ar 16000 -vn "{wav_path}"')
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "opus to wav")
    parser.add_argument('--root_path', type=str, default='/mnt/work4/datasets/keyword/MSWC/mswc_microset', help='root path of dataset directory')
    parser.add_argument('--language', type=str, default='english')
    parser.add_argument('--sample_rate', type=int, default=16000)

    args = parser.parse_args()

    main(args)