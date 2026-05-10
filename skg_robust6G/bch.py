# -*- coding: utf-8 -*-
"""
===============
BCH Codes
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
import operator
import matplotlib.pyplot as plt
from GF_fields_dict_reverse import *
import galois
from sympy import GF, Poly, Pow, Add, Symbol
from sympy import *
import math
from operator import xor
from sympy.abc import x, alpha
from sympy import Matrix, degree_list



__author__ = "Amitha Mayya"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Amitha Mayya, Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.2.7"
__maintainer__ = "Amitha Mayya"
__email__ = "amitha.mayya@barkhauseninstitut.org"
__status__ = "Prototype"

class BCHCodes:
    """A class to generate syndrome and syndrome decoder for BCH codes with Slepian-Wolf implementation"""
    def __init__(self, 
                info_recon: InformationReconciliation = None,
                *args,
                **kwargs) ->None:

        self.info_recon = info_recon
        self.dict_lookup = { 15 : GF_16_binary_power_reverse,
                    31 : GF_32_binary_power_reverse,
                    63 : GF_64_binary_power_reverse,
                    127 : GF_128_binary_power_reverse,
                    255: GF_256_binary_power_reverse,
                    }  
        """
        Args:
        info_recon : The handle to the information reconciliation class which contains the codewords, coderate and type of the code   
        """
    def calculate_syndrome(self, message, t, dict, GF_field, messageR):
        """Calculate the syndrome from message data
            message = received codeword
            t = no. of error correcting capability of the code
            dict = Galois Field dictionary
            GF_field = of the form 2^m -1, corresponds to the field element corresponding to 1 i.e alpha^0
            
            Returns 
            syndrome_power array of the degrees of the syndromes"""
        
        """Syndrome generation reference in Page 4 of M. Walters and S. S. Roy, "Constant-Time BCH Error-Correcting Code," 
        2020 IEEE International Symposium on Circuits and Systems (ISCAS), 2020, pp. 1-5, doi: 10.1109/ISCAS45731.2020.9180846."""
        message_coeffs = message.all_coeffs()
        message_coeffs = list(messageR) #############################################
        message_index = np.flip(np.arange(len(message_coeffs)))
        message_index = (np.arange(len(message_coeffs))) ##########################################################
        message_array = []
        for idx,i in enumerate(message_coeffs):
            if (i == 0):
                message_array.append('zero')
            elif (i == 1):
                message_array.append(message_index[idx])
        syndrome_power = []
        
        for alpha in range(1, (2*t+1)):
            sum = np.zeros(int(np.log2(GF_field+1))).astype(int)
            message_alpha =  []
            for i in range(len(message_array)):
                if(message_array[i] == 'zero'):
                    message_alpha.append('zero')
                else:
                    message_alpha.append((message_array[i] * alpha) % GF_field)
            for j in message_alpha:
                pow_binary = dict[j]
                sum = sum ^ np.array(pow_binary)
            syndrome_power.append(self.get_dict_values(dict,sum))
        return syndrome_power
    
    def bch_decoder(self, message, message_len):
        """Calculates the message length and error correction capability of the received codeword based on the coderate"""
        n = len(message)

        """the following parts are implemented from the following library.
        Hostetter, M. (2020). Galois: A performant NumPy extension for Galois fields [Computer software]. https://github.com/mhostetter/galois"""
        valid_bch_codes = galois.bch_valid_codes(n)
        for i in valid_bch_codes:
            if i[1] <= np.ceil(message_len):
                bch_code = i
                break
        k = bch_code[1]
        print("Message Length",k)
        t = galois.BCH(n,k).t
        print("Errors that can be corrected",t)

        
        message_poly = Poly(message,x).set_domain('ZZ')
        
        #Set the corresponding GF dictionary
        dict_GF_Field = self.dict_lookup[n]

        syndrome_power = self.calculate_syndrome(message_poly, t, dict_GF_Field, n, message) 
        return syndrome_power


    def calculate_discrepancy(self,syndrome_expr,c_expr, GF_field, L,k):
        
        """Calculates the discrepancy for berlekamp massey algorithm
        Syndrome_expr = Received cdeword syndromes
        c_expr = Connection polynomial coeffecients
        GF_field = Galois field
        L = Length of LSFR register
        k = iteration count
        
        Returns
        discrepancy or error in the current syndrome"""
        
        sum = np.zeros((int(np.log2(GF_field + 1))))
        prod_store = []
        c_expr_test = c_expr[:-1] ### ??????????????????????????????????????????????????????????????????????????????

        c_expr_testr = list(reversed(c_expr_test))

        dict = self.dict_lookup[GF_field]

        #Calculates Summation(C_i * S_(k-i)) for i in range (1,L)
        
        for i in range(L):
            curr_syndrome = syndrome_expr[k-i-1]
            curr_syndrome = Poly(curr_syndrome,alpha)
            curr_c = c_expr_testr[i]
            prod = curr_syndrome * curr_c
            prod = prod.degree() % (GF_field)
            prod_store.append(prod)

        ################################
        if nan in prod_store:
            indices1 = [i for i, x in enumerate(prod_store) if x == nan]
            prod_store = [j for i, j in enumerate(prod_store) if i not in indices1]



        #Calculate S_k +  Summation for the discrepancy error"""
        curr_S = syndrome_expr[k]         
        discrepancy_eqn = curr_S
        dis_eqn_poly = Poly(discrepancy_eqn,alpha) 
        degree_de = dis_eqn_poly.degree()
        if (degree_de == -oo):
            degree_de = 'zero'
        degree_de_bin = dict[degree_de]
        gf_sum = np.array(degree_de_bin)

        #Adds the co-effecients in GF field
        for i in range(len(prod_store)):
            # curr_degree = Poly((alpha**prod_store[i]),alpha).degree()
            # degree_binary = dict[curr_degree]
            degree_binary = dict[prod_store[i]]
            gf_sum = gf_sum ^ np.array(degree_binary)
        
        degree_sum = self.get_dict_values(dict, gf_sum)
        
        #Convert into polynomial
        if (degree_sum == 'zero'):
            discrepancy_eqn = 0
        elif (degree_sum == 0):                              
            discrepancy_eqn = Poly((alpha**0),alpha) 
        else:
            discrepancy_eqn = Poly((alpha**degree_sum),alpha)
        
        return discrepancy_eqn

    def get_dict_values(self,dict,binary):

        """Retrives the corresponding binary representation for the given power from the dictionary"""
        
        for k,v in dict.items():
            if(np.array(v)== binary).all():
                power = k
                break
        return power


    def find_gf_inv(self, pow, GF_field):

        """Computes the inverse of the Galois field"""

        inverse = (GF_field - pow) % GF_field
        return inverse
        
    def coeff_addition(self,coeffs, GF_field, dict):
        
        """Checks if the polynomial co-effecients has terms of alpha of different powers and then simplifies it by the GF field addition"""
        
        for idx, coeff in enumerate(coeffs):
            if(coeff.has(Add)):
                coeff_args = coeff.args
                sum_coeffs = np.zeros(int(np.log2(GF_field+1))).astype(int)
                for j in coeff_args:
                    degree = Poly(j,alpha).degree()
                    degree_binary = dict[degree]
                    sum_coeffs = sum_coeffs ^ degree_binary
                degree_pow = self.get_dict_values(dict, sum_coeffs)
                coeffs[idx] = alpha ** degree_pow
            elif (coeff != 0):
                coeff_degree = Poly(coeff,alpha).degree() % GF_field
                coeffs[idx] = alpha ** coeff_degree
        
        return coeffs
        
    def calculate_c(self,d,dm,l,p,c,GF_field):
        
        """Calculates the error locator polynomial of the equation c(x) = c(x) - d*dm_inv*p(x)*x^l"""

        dict = self.dict_lookup[GF_field]
        if(dm == 1):
            pow = 0
        else:
            pow = dm.degree()

        dm_inv = self.find_gf_inv(pow,GF_field)
        dm_inv = Poly((alpha**dm_inv),alpha)
        ddm_inv = (d * dm_inv).degree() % GF_field

        prod = Poly((p*x**l),x)
        prod = Poly(((alpha **ddm_inv)*prod),x) ###########################?

        prod_coeffs = self.coeff_addition(prod.all_coeffs(),GF_field, dict) ##################################?
        prod = Poly.from_list(prod_coeffs, gens=x)
        c = Poly((c-prod),x) ################################################################# should be minus ??
        
        c_coeffs = c.all_coeffs()
        c_coeffs = self.coeff_addition(c_coeffs, GF_field, dict)
        c = Poly.from_list(c_coeffs, gens=x)

        return c

    def berlekamp_massey(self,syndrome_expr, GF_field):

        """Implements berlekamp Massey algorithm to decode the BCH codeword as in 
            https://users.encs.concordia.ca/~msoleyma/ELEC464/ELEC_464_2019/RS-Decoding.pdf

        Inputs:
        syndrome_expr =  Received codeword syndrome
        GF_field = Galois field of the codeword
        
        Outputs:
        c = error locator polynomial
        c_coeffs = co-effecients of the error locator polynomial"""
        # GF_field=15 ########################################################################################
        L = 0                       #Length of LFSR
                
        l = 1                       #k-m, amount of shift update
        
        dm = 1      
        N = len(syndrome_expr)
        c = x ** 0                 # Current Error locator polynomial
        c_coeffs = Poly(c,x).all_coeffs()
        p = x ** 0                 #Error correction polynomial previous value
    
        
        for k in range(0,N,2):
            
            d = self.calculate_discrepancy(syndrome_expr,c_coeffs,GF_field,L,k)
            
            if(d == 0):
                d_coeff = 'zero' ############################################
            else:
                d_coeff = d.degree()
            
            if (d_coeff == 'zero'): ##########################################
                l = l+1          #No change in polynomial
            else:
                if(2*L >= (k+1)):
                    c = self.calculate_c(d,dm,l,p,c,GF_field)      #No length change in update
                    c_coeffs = Poly(c,x).all_coeffs()
                    l = l+1
                
                else:
                    t = c                                      #temporary storage
                    c = self.calculate_c(d,dm,l,p,c,GF_field)      #Update c with length change
                    c_coeffs = Poly(c,x).all_coeffs()
                    L = k - L +1
                    p = t
                    dm = d
                    l = 1
            l = l+1
        return c, c_coeffs


    def chien_search (self,error_locator_coeffs, GF_field):

        """Chien Search to find the roots of error locator polynomial
        
        Inputs: 
        error_locator_coeffs = Co-effecients of the error locator polynomial
        GF_field = Galois field
        
        Outputs:
        error_locator_roots, index = Roots of the polynomial and corresponding index"""

        t = len(error_locator_coeffs)
        """ error_locator_degree = []
        
        for i in error_locator_coeffs:
            coeff_degree = Poly(i,alpha).degree()
            if (coeff_degree == -oo):
                coeff_degree = 'zero'
            error_locator_degree.append(coeff_degree) """
        
        error_locator_degree = [Poly(i,alpha).degree() for i in error_locator_coeffs]
        
        check = None
        # for i in range(error_locator_degree.count(-oo))
        if -oo in error_locator_degree:
            indices = [i for i, x in enumerate(error_locator_degree) if x == -oo]
            check = error_locator_degree.index(-oo) ###################################added
            error_locator_degree = [j for i, j in enumerate(error_locator_degree) if i not in indices]
        ##########################################################################################################################
        error_poly_array = np.flip(np.arange(t)) 
        if check!= None:
            error_poly_array = np.delete(error_poly_array,indices,0) ################# added
        ##########################################################################################################################
        # pow_binary = np.zeros((t))
        binary_len = int(np.log2(GF_field+1))
        
        error_locator_roots = []
        error_locator_index = []
        dict = self.dict_lookup[GF_field]
        for i in range(GF_field):
            sum = np.zeros((binary_len)).astype(int)
            alpha_i = error_poly_array * i
            pow_mul = (alpha_i + np.array(error_locator_degree)) % GF_field
            for j in range(len(pow_mul)):
                pow_binary = dict[pow_mul[j]]
                sum = sum ^ np.array(pow_binary)
            if(sum == 0).all():
                error_locator_index.append(i)
                error_locator_roots.append(self.find_gf_inv(i,GF_field))
        return error_locator_roots, error_locator_index


    def reconciliation_bch_codes(self):

        #BCH code word length = valid power of two -1 
        self.info_recon.code_one = self.info_recon.code_one[:-1] 
        self.info_recon.code_two = self.info_recon.code_two[:-1]
        
        #Determine the Galois field depending on the codeword length
        GF_field = len(self.info_recon.code_one) 
        message_len = len(self.info_recon.code_one) * self.info_recon.code_rate
        dict = self.dict_lookup[GF_field]
    
        #Generate the syndromes
        syndrome_power_one = self.bch_decoder(self.info_recon.code_one, message_len)
        syndrome_power_two = self.bch_decoder(self.info_recon.code_two, message_len)

        syndrome_binary_one=  [dict[syndrome] for syndrome in syndrome_power_one]
        syndrome_binary_two =  [dict[syndrome] for syndrome in syndrome_power_two]
        syndrome_xor_power = []
        for i in range(len(syndrome_binary_one)):
            syndrome_xor = np.array(syndrome_binary_one[i]) ^ np.array(syndrome_binary_two[i])
            syndrome_power = self.get_dict_values(dict,syndrome_xor)
            syndrome_xor_power.append(syndrome_power) 
        # print("XOR Syndrome",syndrome_xor_power)
        syndrome_expr = []
        
        for i in syndrome_xor_power:
            if (i == 'zero'):
                syndrome_expr.append(0)
            else:
                syndrome_expr.append(alpha ** i)

        #Decode the syndrome
        c,coeffs = self.berlekamp_massey(syndrome_expr, GF_field)
        #Solve the polynomial to find the roots
        error_locations, index = self.chien_search((coeffs),GF_field)
        return error_locations

    
    def SW_Error_Correction(self) -> Tuple[bool, np.ndarray, np.ndarray, int]:
        """Implements Error Correction using Slepian Wolf algorithm and Polar Codes for the channel codes
        Returns:
            
            recon_code_one (np.ndarray): Reconciliated code after Slepian-Wolf for code one
            recon_code_two (np.ndarray): Reconciliated code after Slepian-Wolf for code two
            
        """
        
        error_locations = self.reconciliation_bch_codes()
         
        recon_code_one = self.info_recon.code_one
        recon_code_two = self.info_recon.code_two
        recon_code_two[error_locations] = 1-recon_code_two[error_locations]

        return  recon_code_one, recon_code_two

    
         