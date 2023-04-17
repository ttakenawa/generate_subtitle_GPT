import streamlit as st
import ffmpeg
import zipfile
import utils
import os
import tempfile



st.title('Audio to translation subtitle generation')



markdown = ''' 
**An application that uses OpenAI APIs to generate Japanese subtitles, 
English subtitles and a summary from Japanese audio or video files.**

**About this application**

- You will need an Open AI API key. You can get your key [here](https://platform.openai.com/).
- We use Whisper model (large-v2) for transcription and Chat GPT (gpt-3.5-turbo) for translation and summarization.
- The price will be around $ 0.15 for 20 minutes of audio.

**Data policy:**

- The data you upload will be sent to Open AI.
- It will not be used to train models but may be viewed by Open AI.
- This application does not retain any data after your session.
'''
st.markdown(markdown)
st.header('Setting')
api_key = st.text_input('Enter your OpenAI API key (do not include \' or " ).', '', type="default")
st.write('Your API key is', api_key,)

markdown = '''
**Enter up to about 10 Japanese fields or technical terms by modifying the form below.**

This is not required, but will improve the quality of the transcription.
'''

st.markdown(markdown)
terms0 = st.text_input('Terms0', '機械学習、数式、確率論、統計、微分、 データサイエンス', type="default")
terms1 = st.text_input('Terms1', '根元事象、NumPy（ナムパイ）、Python（パイソン）、TensorFlow（テンサーフロー）、PyTorch（パイトーチ）', type="default")
talk_type = st.selectbox('What kind of talk is your audio?',
                         ('Lecture', 'Meeting', 'Conversation'))

if talk_type == 'Lecture': ttype = '準備は良いですか。それでは授業を開始します。'
if talk_type == 'Meeting': ttype = '準備は良いですか。それでは会議を開始します。'
if talk_type == 'Conversation': ttype = 'ご機嫌いかがですか。それではお話ししましょう。' 
prompt = '以下の内容を含みます：'+ terms0 + '、' + terms1 + '。\n ' + ttype

show_summary = st.checkbox('Check the box if you want an English summary')
summarize_ratio = st.number_input('Summarization ratio from 0.1 to 1.0 (only valid if you check the above box.)', min_value=0.1, max_value=1.0, value=0.2)

uploaded_file = st.file_uploader("**Upload a MP3 or a MP4 file.**", type=["mp3", "mp4"])

test_mode = st.checkbox('**Test mode:** check the box if you want to execute **only for the first 120 seconds.**')


st.write('**Click the following Execute button to generate transcriptions and translations, and then download a zip file.**')
execute = st.button('Execute')

with tempfile.TemporaryDirectory(prefix="tmp_", dir=".") as dirpath:
    print(dirpath)  # ./tmp_xo0tg2u1


    if execute:
        if api_key == '': 
            st.write('Enter your API key')        
        elif uploaded_file is None: 
            st.write('Upload MP3 or MP4 file')
        elif not (uploaded_file.name[-4:] in ['.mp3', '.mp4']):
            st.write('Upload MP3 or MP4 file whose file name ends with .mp3 or .mp4') 

        else:
            file_name = uploaded_file.name

            st.write('Preparing files...')
            if file_name[-4:] == ".mp3":
                    
                # Set the path to the MP3 file
                mp3_file_path = os.path.join(dirpath,  file_name)
                with open(mp3_file_path,"wb") as f:
                        f.write(uploaded_file.getvalue())
                
                # test mode
                if test_mode:
                    os.rename(mp3_file_path, mp3_file_path[:-4] + '_temp.mp3')
                    stream = ffmpeg.input(mp3_file_path[:-4] + '_temp.mp3') 
                    stream = ffmpeg.output(stream, mp3_file_path, t=120, ss=0)
                    # 実行 
                    ffmpeg.run(stream, overwrite_output=True) 

            
            if file_name[-4:] == ".mp4":
                    
                # Set the path to the MP4 file
                mp4_file_path =  os.path.join(dirpath,  file_name)
                with open(mp4_file_path,"wb") as f:
                        f.write(uploaded_file.getvalue())


                mp3_file_path = mp4_file_path[:-3] + "mp3"

                # Convert MP4 to MP3
                stream = ffmpeg.input(mp4_file_path) 
                if test_mode: 
                    stream = ffmpeg.output(stream, mp3_file_path, t=120, ss=0) 
                else:
                    stream = ffmpeg.output(stream, mp3_file_path)  
                ffmpeg.run(stream, overwrite_output=True)
            
            
            info = ffmpeg.probe(mp3_file_path)
            duration = float(info['streams'][0]['duration']) # unit [s]
            file_size = float(info['format']['size'])/1e6 # unit [M]
            
            # split the MP3 file with overlaps if its size is more than 24 [M]
            max_file_size = 24  #Default 24
            split_num = int(file_size//max_file_size) + 1
            each_duration = duration/split_num  
            print(each_duration)
            overlap = 20.0

            if split_num > 1:
                for i in range(split_num):
                    stream = ffmpeg.input(mp3_file_path) 
                    # 出力 
                    stream = ffmpeg.output(stream, mp3_file_path[:-4] + str(i) + '.mp3', t=each_duration + overlap, ss= each_duration * i)
                    # 実行 
                    ffmpeg.run(stream, overwrite_output=True) #[M]
            
            st.write('Transcribing ...')
            total_duration, response_list = utils.get_transcribe(mp3_file_path, split_num, api_key, prompt)
            st.write('Transcription completed. The total duration was', '{:.1f}'.format(total_duration/60) , 'min. The cost was about $', '{:.3f}'.format(total_duration /60 * 0.006))

            st.write('Translating ...')
            texts, starts, ends = utils.get_textlists(response_list, split_num, each_duration)
            start_times, end_times, lines_ja = utils.make_sentenses(starts, ends, texts)
            total_token, text_en = utils.get_translation(lines_ja, api_key)
            st.write('Translation completed. The number of token was ', total_token, ' tokens. The cost was about $', '{:.3f}'.format(total_token * 0.002 / 1000))
            lines_en = utils.text2list(text_en)

            ja_srt_data = utils.make_srt(start_times, end_times, lines_ja, mp3_file_path, language='ja') 
            ja_csv_data = utils.make_csv(start_times, end_times, lines_ja, mp3_file_path, language='ja')
            en_srt_data = utils.make_srt(start_times, end_times, lines_en, mp3_file_path, language='en') 
            en_csv_data = utils.make_csv(start_times, end_times, lines_en, mp3_file_path, language='en')

            if show_summary:
                st.write('Summarizing ...')
                total_token, summary_en = utils.get_summary(lines_en, api_key, summarize_ratio = summarize_ratio)
                st.write('Summarization completed. The cost was about $', '{:.3f}'.format(total_token * 0.002 / 1000))
                output_file_path = mp3_file_path[:-4] + '_summary.txt'

                with open(output_file_path, 'w', encoding='utf-8') as f:
                    f.write(summary_en)
                f.close()
            
            zip_file_path = mp3_file_path[:-4] + '.zip'
            with zipfile.ZipFile(zip_file_path, 'w') as zip:
                arcname = os.path.basename(zip_file_path)[:-4]
                zip.write(mp3_file_path[:-4] + '_ja.srt', arcname + '_ja.srt')
                zip.write(mp3_file_path[:-4] + '_ja.csv', arcname + '_ja.csv')
                zip.write(mp3_file_path[:-4] + '_en.srt', arcname + '_en.srt')
                zip.write(mp3_file_path[:-4] + '_en.csv', arcname + '_en.csv')
                if show_summary:
                    zip.write(mp3_file_path[:-4] + '_summary.txt', arcname + '_summary.txt')        
            zip.close()
            

            with open(zip_file_path, "rb") as file:
                btn = st.download_button(
                    label = "Download created files",
                    data = file,
                    file_name = os.path.basename(zip_file_path),
                    mime= "application/zip"
                )

