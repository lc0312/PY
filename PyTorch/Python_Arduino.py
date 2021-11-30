# -*- coding: utf-8 -*-
"""
Spyder Editor
Dixon001
This is a temporary script file.
"""

import time 
import keyboard
from pymata4 import pymata4

board = pymata4.Pymata4() # setting board

class piezo:
    board.set_pin_mode_pwm_output (11) #set pwm output from pins
    board.set_pin_mode_pwm_output (10)
    board.set_pin_mode_pwm_output (6)
    board.set_pin_mode_pwm_output (5)
    
    def d_11_func (x) :
        
        board.pwm_write(11, 16) #set pin 11 output voltage (0-255),here is 16,larger number larger stepsize
        time.sleep(0.01)
        board.pwm_write(11, 0) #set pin 11 shut down

        
    def d_10_func (x) :
    
        board.pwm_write(10, 16)
        time.sleep(0.01)
        board.pwm_write(10, 0)

        
    def d_6_func (x) :
    
        board.pwm_write(6, 32)
        time.sleep(0.01)
        board.pwm_write(6, 0)

        
    def d_5_func (x) :
    
        board.pwm_write(5, 32)
        time.sleep(0.01)
        board.pwm_write(5, 0)
        
class stepper_motor:
      
    board.set_pin_mode_stepper(8, [13,12,8,7]) # set  is 8, stepper connected to pins 13,12,8,7
    
    def stepper_forward (x):
        
        board.stepper_write(12, 1) # move at speed 12, 1 step forward
        board.stepper_write(0, 0) # stop the stepper
        
    def stepper_backward (x):
        
        board.stepper_write(12, -1) # move at speed 12, 1 step backward
        board.stepper_write(0, 0)
        

def reset_board (x):
    board.reset_board ()

   
    
keyboard.on_press_key("+", stepper_motor.stepper_forward) 
keyboard.on_press_key("-", stepper_motor.stepper_backward)
keyboard.on_press_key("9", piezo.d_11_func) 
keyboard.on_press_key("1", piezo.d_10_func) 
keyboard.on_press_key("7", piezo.d_6_func) 
keyboard.on_press_key("3", piezo.d_5_func)
keyboard.on_press_key("5", reset_board)

