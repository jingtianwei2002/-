import os
import time
import threading
import traceback
import re
import numpy as np
import pyaudio
from pymongo import MongoClient, ASCENDING
from funasr import AutoModel
from RealtimeTTS import TextToAudioStream, SystemEngine, ElevenlabsEngine, AzureEngine
import openai
from tenacity import retry, stop_after_attempt, wait_fixed
from multiprocessing import Process

import numpy as np
import librosa
import soundfile as sf

import dashscope
from dashscope.audio.tts_v2 import *


# Disable tqdm progress bars globally
os.environ["TQDM_DISABLE"] = "1"

# MongoDB Configuration
MONGODB_URI = "localhost"
MONGODB_PORT = 27017
DATABASE_NAME = 'jarvis'
COLLECTION_NAME = 'audio_chat'

# Audio Configuration
AUDIO_CHANNELS = 1
CHUNK_SIZE = [0, 10, 5]  # 600ms

# ASR Model Configuration
ASR_MODEL = "D:/Code/jarvis/pretrainmodel"  # 使用绝对路径
MICROPHONE_DEVICE_INDEX = 1 # Device 3: VoiceMeeter VAIO3 Output (VB-Au - Input Channels: 8
OUTPUT_DEVICE_INDEX = 7 # Device 8: VoiceMeeter Aux Input (VB-Audio - Input Channels: 0

# LLM Configuration
LLM_API_KEY = "sk-e745791d41df4624ab1552b72402e49c"
LLM_BASE_URL = "https://api.deepseek.com/v1"
PREPROCESS_LLM = 'deepseek-chat'
ROLEPLAY_LLM = 'deepseek-chat'

TTS_PROVIDER = "alibaba"
# Azure TTS Configuration


# Alibaba TTS Configuration
dashscope.api_key = "sk-6b77887f54db4266bfc4efe523cdd76e"
model = "cosyvoice-v1"
voice = "longxiaochun"

CONVERSATION_MODE = "group_chat" # "group_chat" or "single_chat"

# Group Chat Configuration
ADAPTIVE_RESPONSE_PROMPT = "You are a conversation status evaluator. an AI assistant named ‘小明’ is participating in a group discussion and needs to judge whether to speak. Please judge based on the conversation flow. If the assistant need to respond, reply YES, otherwise NO. Don't Response any other characteristics except YES OR NO."


CONVERSATION_PROACTIVITY = "adaptive"


class MongoDBHandler:
    def __init__(self, uri, port, db_name, collection_name):
        try:
            self.client = MongoClient(uri, port)
            self.db = self.client[db_name]
            self.collection = self.db[collection_name]
        except Exception as e:
            print(f"Failed to connect to MongoDB: {e}")
            raise

    def initialize_database(self):
        self.db.drop_collection(COLLECTION_NAME)
        self.collection.create_index([('timestamp', ASCENDING)])
        self.collection.insert_one({'timestamp': time.time(), 'speaking_status': 1, 'user_speaking_status': 0, 'assistant_speaking_status': 0})
        self.collection.insert_one({'timestamp': time.time(), 'type': 'conversation_history', 'content': []})
        self.collection.insert_one({'timestamp': time.time(), 'type': 'conversation_state', 'params': {
            'conversation_mode': CONVERSATION_MODE,
            'conversation_proactivity': CONVERSATION_PROACTIVITY,
            'adaptive_response_prompt': ADAPTIVE_RESPONSE_PROMPT,
            'voice': voice,
            'silent_threshold': 2.0
        }})

    def update_speaking_status(self, user_status, assistant_status):
        self.collection.find_one_and_update({'speaking_status': 1}, {'$set': {'user_speaking_status': user_status, 'assistant_speaking_status': assistant_status}})

    def update_conversation_state(self, key, value):
        self.collection.find_one_and_update({'type': 'conversation_state'}, {'$set': {f'params.{key}': value}})

    def get_conversation_state(self):
        return self.collection.find_one({'type': 'conversation_state'})

    def get_conversation_history(self):
        return self.collection.find_one({'type': 'conversation_history'})
    
    def update_conversation_history(self, conversation_history):
        merged_conversation = []
        current_entry = conversation_history[0]

        for entry in conversation_history[1:]:
            if entry['role'] == current_entry['role']:
                current_entry['content'] += ' ' + entry['content']
            else:
                merged_conversation.append(current_entry)
                current_entry = entry

        # Append the last accumulated entry
        merged_conversation.append(current_entry)
        self.collection.find_one_and_update({'type': 'conversation_history'}, {'$set': {'content': merged_conversation}})

    def insert_new_text_chunk(self, text, role):
        self.collection.insert_one({
            'timestamp': time.time(),
            'is_processed': 0,
            'type': 'chat_text_chunks',
            'role': role,
            'content': text
        })

    def get_speaking_status(self):
        return self.collection.find_one({'speaking_status': 1})

    def get_unprocessed_text_chunks(self, role):
        return self.collection.find({'is_processed': 0, 'type': 'chat_text_chunks', 'role': role}).sort('timestamp', 1)

    def mark_text_chunks_as_processed(self, message_id):
        self.collection.find_one_and_update({'_id': message_id}, {'$set': {'is_processed': 1}})

    def delete_unprocessed_text_chunks(self, role):
        self.collection.delete_many({'is_processed': 0, 'type': 'chat_text_chunks', 'role': role})


class Callback(ResultCallback):
    _player = None
    _stream = None

    def on_open(self):
        # print("websocket is open.")
        self._player = pyaudio.PyAudio()
        self._stream = self._player.open(
            format=pyaudio.paInt16, channels=1, output_device_index=OUTPUT_DEVICE_INDEX, rate=22050, output=True
        )

    def on_complete(self):
        
        print("speech synthesis task complete successfully.")

    def on_error(self, message: str):
        print(f"speech synthesis task failed, {message}")

    def on_close(self):
        pass
        # print("websocket is closed.")
        # 停止播放器
        # self._stream.stop_stream()
        # self._stream.close()
        # self._player.terminate()

    def on_event(self, message):
        pass
        # print(f"recv speech synthsis message {message}")

    def on_data(self, data: bytes) -> None:
        # print("audio result length:", len(data))
        self._stream.write(data)


class AudioManager:
    def __init__(self, db_handler):
        self.db_handler = db_handler
        self.py_audio = pyaudio.PyAudio()
        self.chunk_stride = CHUNK_SIZE[1] * 960
        self.cache = {}
        self.last_text_time = time.time()
        self.tts_provider = TTS_PROVIDER
        self.silence_threshold = 2.0
        self.tts_text_buffer = ""
        self.filter_words = [
            "嗯", "唉", "啊", "哦", "呃", "哎", "哼", "哈", "嘿", "哇", "呀", "哟", "呦", "呜", "么", "吧",
            "um", "uh", "ah", "oh", "er", "eh", "hmm", "ha", "wow", "yeah", "yup"
        ]
        
        # ASR setup
        try:
            self.asr_model = AutoModel(
                model=ASR_MODEL,
                model_revision="v2.0.4",
                disable_update=True  # 禁用更新检查
            )
        except Exception as e:
            print(f"ASR 模型初始化失败: {e}")
            print(f"模型路径: {ASR_MODEL}")
            raise
        # Audio streams
        self.input_stream = None
        self.is_running = False

        # Track speech and silence durations
        self.speech_durations = []
        self.silence_durations = []
        self.silence_start_time = None

    def start(self):
        """Start both ASR and TTS processing"""
        self.is_running = True
        self.input_stream = self.py_audio.open(
            format=pyaudio.paInt16,
            channels=AUDIO_CHANNELS,
            rate=16000,
            input=True,
            frames_per_buffer=self.chunk_stride,
            input_device_index=MICROPHONE_DEVICE_INDEX
        )

        # Start ASR processing thread
        self.asr_thread = threading.Thread(target=self._asr_processing_loop)
        self.asr_thread.start()

        # Start TTS monitoring thread
        self.tts_thread = threading.Thread(target=self._tts_monitoring_loop)
        self.tts_thread.start()

    def stop(self):
        """Stop all audio processing"""
        self.is_running = False
        if self.input_stream:
            self.input_stream.stop_stream()
            self.input_stream.close()
        self.py_audio.terminate()
        

    def _asr_processing_loop(self):
        while self.is_running:
            try:
                audio_chunk = self.input_stream.read(self.chunk_stride, exception_on_overflow=False)
                self._process_audio_chunk(audio_chunk)
            except Exception as e:
                print(f"Error in ASR processing: {e}")

    def _process_audio_chunk(self, audio_chunk):
        speech_chunk = np.frombuffer(audio_chunk, dtype=np.int16)
        asr_res = self.asr_model.generate(
            input=speech_chunk,
            cache=self.cache,
            is_final=False,
            chunk_size=CHUNK_SIZE,
            encoder_chunk_look_back=4,
            decoder_chunk_look_back=1
        )

        current_time = time.time()
        if current_time - self.last_text_time > self.silence_threshold:
            self.db_handler.update_speaking_status(user_status=0, assistant_status=0)
            if self.silence_start_time is None:
                self.silence_start_time = current_time
            else:
                self.silence_durations.append(current_time - self.silence_start_time)
                self.silence_start_time = current_time

        if asr_res and isinstance(asr_res, list) and len(asr_res) > 0:
            text = asr_res[0].get('text', '').strip()
            if text and text not in self.filter_words:
                self.db_handler.insert_new_text_chunk(text, 'user')
                self.db_handler.update_speaking_status(user_status=1, assistant_status=0)
                self.last_text_time = current_time
                if self.silence_start_time is not None:
                    self.speech_durations.append(current_time - self.silence_start_time)
                    self.silence_start_time = None

        # Adjust the silence threshold dynamically
        self.adjust_silence_threshold()

    def adjust_silence_threshold(self):
        # Improved version:
        if len(self.silence_durations) > 5:
            recent_silences = self.silence_durations[-5:]
        else:
            recent_silences = self.silence_durations
        
        if recent_silences:
            # Calculate statistics
            avg_silence = np.mean(recent_silences)
            std_silence = np.std(recent_silences)
            
            # Base threshold on mean + standard deviation with wider bounds
            new_threshold = avg_silence + std_silence
            
            # Apply stronger bounds
            MIN_THRESHOLD = 0.5  # Minimum 1 second
            MAX_THRESHOLD = 4.0  # Maximum 4 seconds
            
            # Smooth the transition
            current_threshold = self.silence_threshold
            smoothing_factor = 0.7
            
            new_threshold = max(MIN_THRESHOLD, min(MAX_THRESHOLD, new_threshold))
            self.silence_threshold = (smoothing_factor * current_threshold + 
                                    (1 - smoothing_factor) * new_threshold)
        else:
            self.silence_threshold = 2.0  # Default threshold

    def _tts_monitoring_loop(self):
        """Monitor for new assistant responses to speak"""
        while self.is_running:
            try:
                # Check for new TTS messages in MongoDB
                messages = list(self.db_handler.get_unprocessed_text_chunks('assistant'))
                # print('total messages:', len(messages))
                for message in messages:
                    speaking_status = self.db_handler.get_speaking_status()
                    self.db_handler.mark_text_chunks_as_processed(message['_id'])
                    if speaking_status['user_speaking_status'] == 1:
                        self.db_handler.delete_unprocessed_text_chunks('assistant')
                        break
                    self.speak(message['content'])
                    self.tts_text_buffer += message['content']
                
                if self.tts_text_buffer:
                    conversation_history = self.db_handler.get_conversation_history()['content']
                    conversation_history.append({'role': 'assistant', 'content': self.tts_text_buffer})
                    self.db_handler.update_conversation_history(conversation_history)
                    self.tts_text_buffer = ""
                
                time.sleep(0.1)
            except Exception as e:
                print(f"Error in TTS monitoring loop: {e}")
                traceback.print_exc()

    def speak(self, text):
        """Speak the given text."""
        try:
            callback = Callback()
            self.tts_engine = SpeechSynthesizer(
                model=model,
                voice=voice,
                format=AudioFormat.PCM_22050HZ_MONO_16BIT,
                callback=callback,
            )
            print(f'Silent threshold: {self.silence_threshold}')
            self.tts_engine.streaming_call(text)
            self.tts_engine.async_streaming_complete()

            # Wait for the speech synthesis to complete or an interruption
            while not self.tts_engine.complete_event.is_set():
                # Check for user interruption frequently
                if self.db_handler.get_speaking_status().get('user_speaking_status') == 1:
                    print("User interrupted. Stopping TTS.")
                    self.tts_engine.streaming_cancel()
                    break
                time.sleep(0.05)  # Check status more frequently

        except Exception as e:
            self.tts_engine.call('稍等我想想啊')
            self.tts_engine.streaming_complete()
            # print(f"Error during TTS: {e}")
            # traceback.print_exc()
        finally:
            # Ensure resources are released properly
            if self.tts_engine:
                self.tts_engine.close()

class ResponseHandler:
    def __init__(self, db_handler):
        try:
            self.db_handler = db_handler
            openai.api_key = LLM_API_KEY
            openai.api_base = LLM_BASE_URL
            print("[Debug] ResponseHandler successfully initialized")
        except Exception as e:
            print(f"[Error] Failed to initialize ResponseHandler: {e}")
            traceback.print_exc()
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    def _call_llm(self, messages, stream=False, temperature=0.7):
        """封装 DeepSeek API 调用"""
        try:
            print(f"[Debug] Preparing DeepSeek call with {len(messages)} messages")
            response = openai.ChatCompletion.create(
                model=ROLEPLAY_LLM,
                messages=messages,
                temperature=temperature,
                stream=stream
            )
            return response
        except Exception as e:
            print(f"[Error] DeepSeek API call failed: {e}")
            traceback.print_exc()
            raise

    def generate_response(self):
        try:
            print("[Debug] Starting response generation")
            conversation_history = self.db_handler.get_conversation_history()['content']
            
            if not conversation_history:
                print("[Debug] Empty conversation history")
                return
                
            if not self.should_respond(conversation_history):
                print("[Debug] Decided not to respond")
                return
            
            try:
                response = self._call_llm(conversation_history, stream=True)
                
                # 处理流式响应
                for chunk in response:
                    if chunk.choices and chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        self.db_handler.insert_new_text_chunk(content, 'assistant')
                        if self.db_handler.get_speaking_status()['user_speaking_status'] == 1:
                            print("[Debug] User interrupted. Stopping response generation.")
                            break

            except Exception as e:
                print(f"[Error] Error during response generation: {e}")
                traceback.print_exc()
                self.db_handler.insert_new_text_chunk("抱歉，我需要思考一下。", 'assistant')

        except Exception as e:
            print(f"[Error] Error in generate_response: {e}")
            traceback.print_exc()

    def should_respond(self, conversation_history):
        """判断是否应该响应"""
        try:
            if CONVERSATION_PROACTIVITY == "always_respond":
                return True
            if CONVERSATION_PROACTIVITY == "never_respond":
                return False
            
            print("[Debug] Preparing prompt for response decision")
            messages = [
                {"role": "system", "content": ADAPTIVE_RESPONSE_PROMPT},
                {"role": "user", "content": "\n".join([
                    f"{msg['role']}: {msg['content']}" 
                    for msg in conversation_history[-5:]
                ])}
            ]
            
            try:
                response = self._call_llm(messages, stream=False, temperature=0.1)
                content = response.choices[0].message.content.strip().lower()
                print(f"[Debug] Response decision: {content}")
                return 'yes' in content
                
            except Exception as e:
                print(f"[Error] Failed to get response decision: {e}")
                return True
            
        except Exception as e:
            print(f"[Error] Error in should_respond: {e}")
            traceback.print_exc()
            return True

class ConversationManager:
    def __init__(self, db_handler):
        self.db_handler = db_handler
        self.user_text_buffer = ""
        self.last_process_time = time.time()
        self.punc_model = AutoModel(model="ct-punc", model_revision="v2.0.4")
        self.response_handler = ResponseHandler(db_handler)
        self.MAX_HISTORY_LENGTH = 100  # max number of turns to keep

    def load_system_prompt(self):
        import os
        prompt_path = os.path.join('jarvis', 'audio_chat', 'prompts', 'system_prompt.txt')
        try:
            with open(prompt_path, 'r', encoding='utf8') as f:
                system_prompt = f.read()
            conversation_history = self.db_handler.get_conversation_history()
            conversation_history['content'].append({'role': 'system', 'content': system_prompt})
            self.db_handler.update_conversation_history(conversation_history['content'])
        except FileNotFoundError:
            print(f"系统提示文件不存在: {prompt_path}")
            raise
        except Exception as e:
            print(f"加载系统提示时出错: {e}")
            raise

    def start_conversation(self):
        """Main conversation loop"""
        self.load_system_prompt()
        while True:
            try:
                # Get conversation history
                self.process_messages()
                # conversation_history = self.db_handler.get_conversation_history()['content']
                # print(conversation_history[1:])
                time.sleep(0.1)
            except Exception as e:
                print(f"Error in conversation loop: {e}")
                if isinstance(e, (KeyboardInterrupt, SystemExit)):  # Allow interruption
                    raise
                time.sleep(0.1)

    def process_messages(self):
        """Process incoming messages and generate responses"""

        speaking_status = self.db_handler.get_speaking_status()

        # Handle silence and generate response
        if speaking_status['user_speaking_status'] == 0 and speaking_status['assistant_speaking_status'] == 0:
            self.handle_silence()

    def process_user_input(self):
        unprocessed_user_chunks = self.db_handler.get_unprocessed_text_chunks(role='user')
        new_chunks = ""
        if not unprocessed_user_chunks:
            return
        for chunk in unprocessed_user_chunks:
            new_chunks += chunk['content'] + " "
            self.db_handler.mark_text_chunks_as_processed(chunk['_id'])
        if new_chunks.strip():
            punc_res = self.punc_model.generate(input=new_chunks)
            self.user_text_buffer += punc_res[0]['text']

    def handle_silence(self):
        try:
            self.process_user_input()
            if not self.user_text_buffer or len(self.user_text_buffer) < 2:
                return
          
            conversation_history = self.db_handler.get_conversation_history()['content']
            conversation_history.append({'role': 'user', 'content': self.user_text_buffer})
            self.db_handler.update_conversation_history(conversation_history)
            
            # Generate and process response
            self.response_handler.generate_response()

            # Clear buffer
            self.user_text_buffer = ""
            
        except Exception as e:
            print(f"Error handling silence: {e}")
            traceback.print_exc()

    def trim_conversation_history(self):
        """Maintain conversation history size"""
        conversation_history = self.db_handler.get_conversation_history()['content']
        if len(conversation_history) > self.MAX_HISTORY_LENGTH + 1:  # +1 for system prompt
            conversation_history = (
                [conversation_history[0]] +  # Keep system prompt
                conversation_history[-(self.MAX_HISTORY_LENGTH):]  # Keep recent messages
            )
            self.db_handler.update_conversation_history(conversation_history)



def run_audio_manager():
    """Process for handling audio input/output"""
    db_handler = MongoDBHandler(MONGODB_URI, MONGODB_PORT, DATABASE_NAME, COLLECTION_NAME)
    audio_manager = AudioManager(db_handler)
    
    try:
        audio_manager.start()
        while True:
            time.sleep(0.1)  # Prevent CPU hogging
    except KeyboardInterrupt:
        audio_manager.stop()
    except Exception as e:
        print(f"Error in audio manager process: {e}")
        traceback.print_exc()
    finally:
        audio_manager.stop()

def run_conversation_manager():
    """Process for handling conversation flow"""
    db_handler = MongoDBHandler(MONGODB_URI, MONGODB_PORT, DATABASE_NAME, COLLECTION_NAME)
    conversation_manager = ConversationManager(db_handler)
    
    try:
        conversation_manager.start_conversation()
    except KeyboardInterrupt:
        print("Conversation manager stopped by user")
    except Exception as e:
        print(f"Error in conversation manager process: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    try:
        # Initialize database
        db_handler = MongoDBHandler(MONGODB_URI, MONGODB_PORT, DATABASE_NAME, COLLECTION_NAME)
        db_handler.initialize_database()

        # Create and start processes
        audio_process = Process(target=run_audio_manager)
        conversation_process = Process(target=run_conversation_manager)

        audio_process.start()
        conversation_process.start()

        # Wait for processes to complete
        audio_process.join()
        conversation_process.join()

    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
    except Exception as e:
        print(f"Error in main process: {e}")
        traceback.print_exc()
    finally:
        # Ensure processes are terminated
        if 'audio_process' in locals() and audio_process.is_alive():
            audio_process.terminate()
        if 'conversation_process' in locals() and conversation_process.is_alive():
            conversation_process.terminate()



