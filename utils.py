import requests
import csv
import re
import time 


def transcribe(file_path, output_format, api_key, prompt):
    '''
    Request form of transcription
    '''
    url = 'https://api.openai.com/v1/audio/transcriptions'
    headers = {'Authorization': f'Bearer {api_key}'}
    file = {'file': open(file_path, 'rb')}
    data = {
        # "fileType": 'mp3', #default is wav
        # "diarization": "false",
        #Note: setting this to be true will slow down results.
        #Fewer file types will be accepted when diarization=true
        #"numSpeakers": "1",
        #if using diarization, you can inform the model how many speakers you have
        #if no value set, the model figures out numSpeakers automatically!
        # "url": "URL_OF_STORED_AUDIO_FILE", #can't have both a url and file sent!
        # "language": "ja", #if this isn't set, the model will auto detect language,
        'model': 'whisper-1',
        'response_format': output_format,
        'prompt': prompt
    }
    response = requests.post(url, headers=headers, files=file, data=data)
    return response


def get_transcribe(mp3_file_path, split_num, api_key, prompt):
    '''
    Get response of transcription
    '''
    total_duration = 0.0
    if split_num == 1:
        response = transcribe(mp3_file_path, 'verbose_json', api_key, prompt).json()
        response_list = [response]
        total_duration = response['duration']

    else:
        response_list = []
        for i in range(split_num):
            response = transcribe(mp3_file_path[:-4] + str(i) + '.mp3', 'verbose_json', api_key, prompt).json()
            response_list.append(response)
            total_duration += response['duration']
    return total_duration, response_list


def seconds2SRT(time, start_time = 0.0): # time in seconds
    '''
    Convert seconds to the form of SRT as hh:mm:ss mss 
    '''
    time = time + start_time
    hours = str(int(time//3600))
    minutes = str(int((time % 3600)//60))
    seconds = str(int((time % 60)//1))
    miliseconds = str(int(((time + 1e-9) % 1.0)//0.001))
    return f'{hours.zfill(2)}:{minutes.zfill(2)}:{seconds.zfill(2)},{miliseconds.zfill(3)}'


def get_textlists(response_list, split_num, each_duration):
    '''
    Get a list of sentences from responses
    '''
    start_seconds_list = []
    starts_list = []
    ends_list = []
    texts_list = []
    for i, response in enumerate(response_list):
        start_seconds_list.append([sentense['start'] for sentense in response['segments']])
        starts_list.append([seconds2SRT(sentense['start'], start_time = each_duration * i) for sentense in response['segments']])
        ends_list.append([seconds2SRT(sentense['end'], start_time = each_duration * i) for sentense in response['segments']])
        texts_list.append([sentense['text'] for sentense in response['segments']])
    
    # Delete duplicate TEXT between split tracks
    for i in range(split_num-1):
        del_from = len(start_seconds_list[i])
        for j, start in enumerate(start_seconds_list[i][:-1]):
            if start > each_duration:
                difference = abs(start - start_seconds_list[i+1][1] - each_duration)

                # Delete the part after entering the overlap part.
                del starts_list[i][j:] 
                del ends_list[i][j:] 
                del texts_list[i][j:] 
                
                # Delete the beginning of the next track if the difference is within 2 seconds
                if difference < 2.0:   
                    del starts_list[i+1][:1] 
                    del ends_list[i+1][:1] 
                    del texts_list[i+1][:1]
                continue
    starts = sum(starts_list, []) # [a,b,c] = sum([[a,b], [c]], []) 
    ends = sum(ends_list, [])
    texts = sum(texts_list, [])

    # Corrected end time to avoid time overlap
    ends[:-1] = starts[1:]


    # Concatenate if the same text is consecutive.
    new_starts = starts[:1]
    new_ends = ends[:1]
    new_texts = texts[:1]

    for i in range(1, len(starts)):
        if new_texts[-1] == texts[i]:
            new_ends[-1] = ends[i]
        else:
            new_starts.append(starts[i])
            new_ends.append(ends[i])
            new_texts.append(texts[i])

    starts = new_starts
    ends = new_ends
    texts = new_texts

    return texts, starts, ends


def make_sentenses(starts, ends, texts):
    '''
    combine sentences so that they are not split in the middle
    '''
    start_times = []
    end_times = []
    lines = ['']
    no_punc = False

    for i, (start, end, text) in enumerate(zip(starts, ends, texts)):
        punc = list(re.finditer(r'。', text))

        # If the previous sentence has "。",  add the current start and end times to the list
        if no_punc == False:
            start_times.append(start)
            end_times.append(end)
        # If the previous sentence has no "。", its start time is not changed, 
        # but its end time is amended to current sentence's
        else:
            end_times[-1] = end
        
        # If the current sentence has "。”, the current sentence is added up to the rightmost "。" 
        # to the prevous sentence, and add the rest to the list as a new sentence
        if not (punc == []):
            lines[-1] += text[:punc[-1].start()+1]
            lines.append( text[punc[-1].start()+1:])
            no_punc = False

        # If the current sentence has no "。”, add the current sentence to the previous one
        else:
            lines[-1] += text
            no_punc = True

    # Deleted after rightmost "。" in the last sentence because there is no corresponding timeline.    
    if no_punc == False: 
        lines = lines[:-1]
    return start_times, end_times, lines


def get_translation(lines_ja, api_key):
    requested_times = [time.time() - 61.0] * 3

    h = {  
        'Content-Type': 'application/json',  
        'Authorization': f'Bearer {api_key}' 
    }  
        
    u = 'https://api.openai.com/v1/chat/completions'  

    text = ''
    text_en = ''
    total_token = 0
    for i, line in enumerate(lines_ja):
        text +=  str(i+1) + '. ' + line + '\n '

        # Request in evry 30 sentences
        if ((i+1) % 30 == 0) or (i+1 == len(lines_ja)): 
            message = [{"role": "system", "content": "Translate the following text in brief English line by line. Use we for the first person.\n"},
                        {"role": "user", "content": text}]
            d = {  
                "model": "gpt-3.5-turbo",  
                "messages": message,  
                # "max_tokens": 100,  
                "temperature": 0  
                }
            new_time = time.time()
            if new_time - requested_times[-3] < 61.0:
                time.sleep(61.0 - (new_time - requested_times[-3]))
            requested_times.append(new_time)

            r = requests.post(url=u, headers=h, json=d).json()
            token = r['usage']['total_tokens']
            print('token:', token)
            total_token += token
            text_en += r['choices'][0]['message']['content']
            # Put '\n' to the last row
            if text_en[-2:] != '\n': text_en += '\n'
            text = ''
    return total_token, text_en



def text2list(text_en):
    '''
    Get a list of sentenses from a long text with \n
    '''
    nums = list(re.finditer(r'\n\d+. ', text_en)) # list of positions of "\n"
    first = re.search(r'1. ', text_en) # position of "1."
    lines_en = []
    if len(nums) > 1:
        lines_en.append(text_en[first.end(): nums[0].start()].replace('\n', ''))
        lines_en += [text_en[nums[i].end(): nums[i+1].start()].replace('\n', '')  for i in range(len(nums)-1)]
        lines_en.append(text_en[nums[len(nums)-1].end():].replace('\n', ''))
    return lines_en


def get_summary(lines_en,  api_key, summarize_ratio = 0.1, batch_size = 100): 
    '''
    Make sumamry
    '''

    h = {  
        'Content-Type': 'application/json',  
        'Authorization': f'Bearer {api_key}' 
    }  
        
    u = 'https://api.openai.com/v1/chat/completions'  

    text = ''
    summary_en = ''
    total_token = 0
    requested_times = [time.time()-61.0] * 3

    for i, line in enumerate(lines_en):
        text +=  line

        # Request for batch_size of lines
        if ((i+1) % batch_size == 0) or (i+1 == len(lines_en)):

            message = [{"role": "system", "content": f"Summarize the following text to {int(summarize_ratio * 100)}% length. Use we for the first person.\n"},
                        {"role": "user", "content": text}]
            d = {  
                "model": "gpt-3.5-turbo",  
                "messages": message,  
                # "max_tokens": 100,  
                "temperature": 0  
                }
            
            new_time = time.time()
            if new_time - requested_times[-3] < 61.0:
                print('wait for ready', 61.0 - (new_time - requested_times[-3]), '[s]')
                time.sleep(61.0 - (new_time - requested_times[-3]))
            requested_times.append(new_time)

            r = requests.post(url=u, headers=h, json=d).json()
            token = r['usage']['total_tokens']
            print('token:', token)
            total_token += token
            summary_en += r['choices'][0]['message']['content']
            text = ''
    if len(lines_en) > batch_size:
        message = [{"role": "system", "content": f"Summarize the following text to {summarize_ratio}% length. Use we for the first person.\n"},
                    {"role": "user", "content": summary_en}]
        d = {  
            "model": "gpt-3.5-turbo",  
            "messages": message,  
            # "max_tokens": 100,  
            "temperature": 0  
            }
        
        new_time = time.time()
        if new_time - requested_times[-3] < 61.0:
            print('wait for ready', 61.0 - (new_time - requested_times[-3]), '[s]')
            time.sleep(61.0 - (new_time - requested_times[-3]))
            
        requested_times.append(new_time)

        r = requests.post(url=u, headers=h, json=d).json()
        token = r['usage']['total_tokens']
        print('token:', token)
        total_token += token
        summary_en = r['choices'][0]['message']['content']
    return total_token, summary_en



def make_srt(starts, ends, texts, mp3_file_path, language = 'en'):
    srt = ''
    for i, (start, end, text) in enumerate(zip(starts, ends, texts)):
        srt += str(i+1) + '\n' + start + ' --> ' + end + '\n' +text + '\n\n'
    srt += '\n'
    if language == 'en':
        output_file_name = mp3_file_path[:-4]+'_en.srt'
    else: 
        output_file_name = mp3_file_path[:-4]+'_ja.srt'
    with open(output_file_name, 'w', encoding='utf-8') as f:
        f.write(srt)
    f.close()
    return srt



def make_csv(starts, ends, texts, mp3_file_path, language = 'en'):
    csv_data = [[i+1,start, end, text] for i, (start, end, text) in enumerate(zip(starts, ends, texts))]
    if language == 'en':
        output_file_name = mp3_file_path[:-4]+'_en.csv'
    else: 
        output_file_name = mp3_file_path[:-4]+'_ja.csv'
    with open(output_file_name, 'w' , encoding='utf_8_sig') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(csv_data)
        f.close()
    return csv_data

