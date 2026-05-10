# -*- coding: utf-8 -*-
"""
===============
Information Reconciliation
===============
"""
from __future__ import annotations
from sympy.combinatorics import GrayCode
import numpy as np
import math
from copy import copy
from typing import Optional, TYPE_CHECKING
from scipy import signal, linalg
import cmath
import matplotlib.pyplot as plt
from polar import PolarCodes
from bch import BCHCodes


__author__ = "Amitha Mayya"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Amitha Mayya, Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.2.7"
__maintainer__ = "Amitha Mayya"
__email__ = "amitha.mayya@barkhauseninstitut.org"
__status__ = "Prototype"

class InformationReconciliation:
    """A class to reconciliate the generated keys between two devices with Slepian-Wolf Error Correcting codes"""

    #__code_rate: float #Code rate
    __block_error_code: str #Error correcting block code
    def __init__(self, 
                 block_error_code: str,
                 code_rate: float = 0.1,
                 device_one: Optional [KeyGeneratingModem] = None,
                 device_two: Optional [KeyGeneratingModem] = None,
                 code_one: Optional [np.array] = None,
                 code_two: Optional [np.array] = None,
                 ####################################
                 SNR_dB: Optional [float] = None,
                 ####################################
                 *args,
                 **kwargs) ->None:
        ############################################
        self.SNR_dB = SNR_dB
        ############################################
        self.channel_code_list = ['Polar_SC', 'Polar_CRC', 'BCH']
        self.block_error_code = block_error_code
        self.code_rate = code_rate
        self.device_one = device_one  
        self.device_two = device_two
        self.code_one = code_one
        self.code_two = code_two
        

        """
        Args:
            code_rate (float): Code rate i.e k/n where k is the message bits and n is the codeword length
            channel_code (str): Error Correcting Codes, possible values: Polar and BCH.
            device_one (KeyGeneratingModem): The key generating modem for the first device, considered as syndrome transmitting modem
            device_two (KeyGeneratingModem): The key generating modem for the second device, considered as syndrome receiving modem
            code_one (np.array): First error correcting code in case device_one is None
            code_two (np.array): Second error correcting code in case device_two is None
        """
    #################################################################################
    @property
    def SNR_dB (self) -> float:
        """Signal to Noise Ratio in dB
        Returns:
            float:
                Signal to Noise Ratio in dB     
        """
        return self.__SNR_dB

    @SNR_dB.setter
    def SNR_dB (self, value:float) -> None:
        self.__SNR_dB = value
    ######################################################################################

    @property
    def code_rate (self) -> float:
        """Code rate which is the ratio of information message length / code length
        
        Returns:
            float:
                code rate 
            Raises:
                Value error: If code rate is greater than one and lesser than 0
        """
        return self.__code_rate
    
    @code_rate.setter
    def code_rate (self, value:float) -> None:
        
        if (value <= 0  or value >= 1) :
            raise ValueError("Code rate must be greater than zero and lesser than one")
        
        self.__code_rate = value

    @property
    def channel_code_list (self) -> list:
        """Available block error correcting codes
        
        Returns:
            list['str']:
                Available linear block error correcting codes 
        """
        return self.__channel_code_list
    
    @channel_code_list.setter
    def channel_code_list (self, value:list) -> None:
        
        self.__channel_code_list = value

    @property
    def block_error_code (self) -> str:
        """Block error correcting code
        
        Returns:
            str:
                Linear block error correcting code 
            Raises:
                Value error: If value does not match to the channel code list
        """
        return self.__block_error_code
    
    @block_error_code.setter
    def block_error_code (self, value:str) -> None:
        
        if not isinstance(value, str):
            raise ValueError("Channel code input must be string")
        if value not in self.channel_code_list:
            raise ValueError("Channel Code Type {} not yet implemented", value) 
        
        self.__block_error_code = value
    
    @property
    def code_one (self) -> np.array:
        """First block error correcting code for the slepian wolf algorithm
        Returns:
            np.array:
                First block error correcting code
            Raises:
                Value error if device one is None and code_one is None
        """
        if self.device_one is not None:
            self.__code_one = self.device_one.raw_key
        else:
            self.__code_one = self.__code_one
        return self.__code_one

    @code_one.setter
    def code_one (self, value:np.array) -> None:

        if self.device_one is None:
            if value is None:
                raise ValueError ("Codeword must be input either via device_one or via code_one")

        self.__code_one = value

    @property
    def code_two (self) -> np.array:
        """Second block error correcting code for the slepian wolf algorithm
        Returns:
            np.array:
                Second block error correcting code
            Raises:
                Value error if device two is None and code_two is None
        """
        if self.device_two is not None:
            self.__code_two = self.device_two.raw_key
        else:
            self.__code_two = self.__code_two
        return self.__code_two

    @code_two.setter
    def code_two (self, value:np.array) -> None:

        if self.device_two is None:
            if value is None:
                raise ValueError ("Codeword must be input either via device_one or via code_one")

        
        self.__code_two = value

    @property
    def codeword_length (self) -> int:
        """Returns the codeword length
        Returns:
            int:
                codeword length
        """
        if not (len(self.code_one) == len(self.code_two)):
            raise ValueError("Code one and code two must be of equal length")
        else: 
            self.__codeword_length = len(self.code_one)
        return self.__codeword_length

    @codeword_length.setter
    def codeword_length (self, value:int) -> None:
        
        self.__codeword_length = value

    @property
    def ecc_class(self) -> Union[PolarCodes, BCHCodes]:
        """Returns the ecc_class handle to the corresponding syndrome generation and decoding
        Returns:
            ecc_class : Handle to the corresponding ecc class
        """
        if self.__block_error_code == 'Polar_SC' or 'Polar_CRC':
            self.__ecc_class = PolarCodes(self)
        if self.__block_error_code == 'BCH':
            self.__ecc_class = BCHCodes(self)
        return self.__ecc_class
    
    @ecc_class.setter
    def ecc_class (self, value:Union[PolarCodes, BCHCodes]) -> None:

        self.__ecc_class = value

    @property
    def recon_code_one (self) -> np.ndarray:
        """Returns the reconciliated code corresponding to code_one"""
        return __recon_code_one

    @property
    def recon_code_two (self) -> np.ndarray:
        """Returns the reconciliated code corresponding to code_two"""
        return __recon_code_two

    def implement_SW (self) -> Tuple[bool,np.ndarray,np.ndarray,int]:
        """Implements Slepian-Wolf algorithm to corrects errors between code_one and code_two depending on the code type.
         Returns:
            Success_recon (bool) : Flag indicating reconciliation succeeded or not
            recon_code_one (np.ndarray) : First reconciliated code
            recon_code_two (np.ndarray) : Second reconciliated code
            Nerrs (int) : Number of errors after reconciliation
        """

        recon_code_one, recon_code_two = self.ecc_class.SW_Error_Correction()
        self.__recon_code_one = recon_code_one
        self.__recon_code_two = recon_code_two

        return  recon_code_one, recon_code_two







        
    
    
    




        


    

    


