cd <folder of duration predictor>
python variance_convert.py --in_codes <path of test_soft_parsed.txt> --spkr_folder <path of speaker_vectors> --pitch_dir <path of target pitch contours> --tgt_emo_dir <path of target emo embeddings> --target_pickle <source-reference json file> --output_file <path of codes to be saved>
cd <folder of bigvgan>
python inference_allpred.py --output_dir <where the wav files will be saved> --pitch_folder <path of target pitch contours> --spkr_folder <path of speaker_vectors> --emo_folder <path of target emo embeddings> --target_files <source-reference json file> --checkpoint_file <bigvgan saved checkpoint> --input_code_file <path of codes to be saved> 
cd <evaluation folder>
python get_whisper_asr.py --input_folder <where the wav files will be saved> --out_file <where the transcripts will be saved>
python compute_wer.py --gen_file <where the transcripts will be saved>
python get_emo_acc.py --converted_folder <where the wav files will be saved>
python get_xvectors.py --converted_folder <where the wav files will be saved> --target_folder <where the x-vectors of the converted files will be saved>
python get_speaker_acc.py --xvector_folder <where the x-vectors of the converted files will be saved>
python prepare_tsv.py --input_folder <where the wav files will be saved> --out_file <some reference tsv file>
python stopes/eval/local_prosody/annotate_utterances.py +data_path=<some reference tsv file> +result_path=<path of rate.tsv> +audio_column=audio_path +text_column=text +speech_units=[word,syllable,char,phoneme,vowel] +vad=true +net=true +lang=en +forced_aligner=ctc_wav2vec2-xlsr-multilingual-56
python get_rhythm_stats.py --input_file <path of rate.tsv>
