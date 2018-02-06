#! /sys/env/python3

import aiy.audio
import aiy.cloudspeech
import aiy.voicehat
import socket
import sys


######################################################################
class SentimentClassifierConnection(object):
    '''
    Gets the sentiment (postive or negative) of the utterance from 
    a sentiment classfier server.
    '''
    def __init__(self):
        self.response_greeting = "Hey, I think that was "
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.openSocketToTweetClassifier()
        
    def openSocketToTweetClassifier(self):
        server_address = ('localhost', 10000)
        print('connecting to {} port {}'.format(server_address[0],server_address[1]))
        self.sock.connect(server_address)

    def run(self, utterance):
        print(utterance)
        self.sock.sendall(bytes(utterance,'ascii'))
        sentiment = str(self.sock.recv(1024), 'utf-8')
        print(sentiment)
        return self.response_greeting + sentiment
    
    def close(self):
        self.sock.close()


def main():
    recognizer = aiy.cloudspeech.get_recognizer()
    button = aiy.voicehat.get_button()
    led = aiy.voicehat.get_led()
    sentiment_classifier = SentimentClassifierConnection()
    
    aiy.audio.get_recorder().start()
    while True:
        led.set_state(aiy.voicehat.LED.BLINK)
        print('Press the button and speak')
        button.wait_for_press()
        led.set_state(aiy.voicehat.LED.ON)
        print('Listening...')
        text = recognizer.recognize()
        if not text:
            print('Sorry, I did not hear you.')
        else:
            print('You said "', text, '"')
            if text == 'goodbye':
                sentiment_classifier.close()
                break
            else:
                aiy.audio.say(sentiment_classifier.run(text))


if __name__ == '__main__':
    main()
