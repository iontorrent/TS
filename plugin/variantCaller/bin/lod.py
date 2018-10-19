#!/usr/bin/python
# Copyright (C) 2018 Thermo Fisher Scientific, Inc. All Rights Reserved
import numpy
import sys
import json
import os
from optparse import OptionParser

# ------------------------------------- Useful Utils ---------------------------------------------
"""
# For debug only, not used.
#import scipy.special
def qual_beta(dp, ao, f_cutoff):
    assert(0 < f_cutoff < 1.0)
    a = ao + 1
    b = dp - ao + 1
    beta_integral = scipy.special.betainc(a, b, f_cutoff)
    return linear_to_phread(min(beta_integral, 1.0 - beta_integral))
"""
    
def linear_to_phread(x):
    return -10.0 / numpy.log(10.0) * numpy.log(x)

def phread_to_linear(x):
    return numpy.exp(-0.1 * numpy.log(10.0) * x);

def log_sum_from_log_individual(log_list):
    """
    calculate log(sum(exp(log_list))) with better numerical stability
    """
    log_array = numpy.array(log_list, dtype = float)
    max_log = log_array.max()
    log_array -= max_log
    return max_log + numpy.log(sum(numpy.exp(log_array)))

def log_abs_diff_from_log(log_x, log_y):
    """
    Calculate log(abs(exp(log_x) - exp(log_y)))
    """
    return max(log_x, log_y) + numpy.log(1.0 - numpy.exp(-abs(log_x - log_y)))

def log_beta_gen(dp, end_dp = None):
    """
    [x for x in log_beta_gen(dp, end_dp)] gives [log(scipy.special.beta(ao + 1, dp - ao + 1)) for ao in xrange(end_dp + 1)] 
    """
    dp = int(dp)
    assert(dp > 0)    
    if end_dp is None:
        end_dp = dp
    else:
        end_dp = int(end_dp)
        assert(0 <= end_dp <= dp)
    init_log_beta = -numpy.log(dp + 1.0)
    log_beta = init_log_beta
    yield log_beta
    for ao in xrange(end_dp - 1 if end_dp == dp else end_dp):
        log_beta += (numpy.log(ao + 1.0) - numpy.log(dp - ao))
        yield log_beta
    if end_dp == dp:
        yield init_log_beta


def beta_inc_gen(dp, f_c, ao_max = None):
    """
    [x for x in inc_beta_gen(dp, f_c, ao_max)] gives [scipy.special.betainc(ao + 1, dp - ao + 1, f_c) for ao in xrange(ao_max + 1)]
    """
    if ao_max is None:
        ao_max = dp
    assert(0.0 < f_c < 1.0)
    assert(dp > 0)
    assert(0 <= ao_max <= dp)
    
    log_f_c = numpy.log(f_c)
    log_1_minus_f_c = numpy.log(1.0 - f_c)
    beta_inc = max(1.0 - numpy.exp((dp + 1.0) * log_1_minus_f_c), 0.0)
    yield beta_inc

    for ao, log_beta in enumerate(log_beta_gen(dp, ao_max - 1)):
        beta_inc -= numpy.exp(log_f_c * (ao + 1.0) + log_1_minus_f_c * (dp - ao) - log_beta - numpy.log(ao + 1.0))
        yield max(beta_inc, 0.0)
 
def linear_interpolation(x, x1, y1, x2, y2):
    if x in [x1, x2]:
        return y1 if x == x1 else y2
    assert(min(x1, x2) < x < max(x1, x2))
    alpha = (x - x2) / (x1 - x2)    
    return alpha * y1 + (1.0 - alpha) * y2
    
# ------------------------------------- Useful Utils ---------------------------------------------
    
# ------------------------------------- BinomialUtils ---------------------------------------------
class BinomialUtils:
    """
    Provide methods for dealing with tail of binomial distribution with high numerical accuracy.
    """
    exact_log_factorial = numpy.cumsum([numpy.log(i) if i > 0 else 0.0 for i in xrange(256 + 1)])
    
    @staticmethod
    def log_factorial(x):
        try:
            return BinomialUtils.exact_log_factorial[int(x)]
        except IndexError:
            return BinomialUtils.stirling_names_approx(x)
    
    @staticmethod
    def stirling_names_approx(x):
        """
        Calculate the log factorial using the Stirling-Names formula, which converges to log(factorial(x)) asymptotically.
        The approximation is very good if x > 20.
        """
        x += 1.0
        # 1.8378770664093453 =log(2*pi)
        log_fact_x = 0.5 * (1.8378770664093453 - numpy.log(x)) + (numpy.log(x + x / (12.0 * x * x - 0.1)) - 1.0) * x        
        return log_fact_x
        
    @staticmethod
    def log_n_choose_k(n, k):
        n = int(n)
        k = int(k)
        if k in [0, n]:
            return 0.0
        assert(0 <= k <= n)
        return max(BinomialUtils.log_factorial(n) - (BinomialUtils.log_factorial(k) + BinomialUtils.log_factorial(n - k)), 1.0)
        
    @staticmethod
    def log_binomial_pmf(n, k, log_p, log_q = None):
        """
        calculate P(X = k) where X ~ Binomial(n, p)
        log_q = log(1-p) can be pre-calculated
        """        
        if log_q is None:
            log_q = BinomialUtils.log_complement_from_log_p(log_q)
        return min(BinomialUtils.log_n_choose_k(n, k) + (k * log_p + (n - k) * log_q), 0.0)
        
    @staticmethod
    def log_binomial_cdf(x, n, log_p, log_q = None):
        """
        calculate sum_{x=0}^k P(X = k) where X ~ Binomial(n, p)
        log_q = log(1-p) can be pre-calculated
        """        
        if log_q is None:
            log_q = BinomialUtils.log_complement_from_log_p(log_q)
        log_pmf = [BinomialUtils.log_binomial_pmf(n, k, log_p, log_q) for k in xrange(int(x) + 1)]
        log_sum = log_sum_from_log_individual(log_pmf)
        return min(log_sum, 0.0)
            
    @staticmethod 
    def log_complement_from_log_p(log_p):
        """
        calculate log(q) = log(1.0 - p) from log(p) where 0 <= p <= 1 
        """
        # First handle the singular cases.
        if not (log_p <= 0.0):
            raise(ValueError("log_p must be negative or zero."))
            return None
        elif log_p == 0.0:
            return -float('inf')
        elif not numpy.isfinite(log_p):
            return 0.0
        
        # Now log(p), log(1-p) are not singular
        p = numpy.exp(log_p)
        q = 1.0 - p
        
        if q == 1.0:
           # non-zero p gives q = 1 implies floating error. Use 1st order Taylor approximation to calculate log(q) ~ -p
            return -p
        elif q == 0.0:
           # p != 1 gives q = 0 implies floating error. Use 1st order Taylor approximation to calculate log(q) ~ log(-log(p))
            return numpy.log(-log_p)
        
        return numpy.log(q)
# ------------------------------------- End BinomialUtils ---------------------------------------------

# ------------------------------------- LodManager ---------------------------------------------

class LodManager:
    def __init__(self):
        self.__min_var_coverage = 2
        self.__min_variant_score = linear_to_phread(0.5)
        self.__min_callable_prob = 0.98
        self.__min_allele_freq = 0.0005
        self.__do_smoothing = True

    def do_smoothing(self, flag):
        if flag:
            self.__do_smoothing = True
        else:
            self.__do_smoothing = False
        return self.__do_smoothing
        
    def set_parameters(self, param_dict):
        self.__min_var_coverage = param_dict.get('min_var_coverage', self.__min_var_coverage)
        assert(self.__min_var_coverage >= 0)
        self.__min_variant_score = param_dict.get('min_variant_score', self.__min_variant_score)
        # min_variant_score < 3.010 is equivalent to min_variant_score = 3.010
        self.__min_variant_score = max(self.__min_variant_score, linear_to_phread(0.5))
        self.__min_callable_prob = param_dict.get('min_callable_prob', self.__min_callable_prob)
        assert(0.0 < self.__min_callable_prob < 1.0)
        self.__min_allele_freq = param_dict.get('min_allele_freq', self.__min_allele_freq)
        assert(0.0 < self.__min_allele_freq < 1.0)


    def __min_callable_ao(self, dp):
        if dp == 0 or dp < self.__min_var_coverage:
            return None, None, None
        
        qual_minus = None
        
        for ao, beta_inc in enumerate(beta_inc_gen(dp, self.__min_allele_freq)):
            # I aim at calling variant, so I allow QUAL < 3.010.
            # beta_inc = scipy.special.betainc(ao + 1, dp - ao + 1, self.__min_allele_freq)
            qual = linear_to_phread(beta_inc)

            if qual >= self.__min_variant_score and ao >= self.__min_var_coverage:
                return ao, qual, qual_minus
            qual_minus = qual
            
        return None, None, None

    def __callable_prob(self, dp, af, min_callable_ao, qual_plus = None, qual_minus = None):
        assert(0.0 <= af <= 1.0)
        log_p = numpy.log(af)
        log_q = BinomialUtils.log_complement_from_log_p(log_p)
        log_pmf_list = [BinomialUtils.log_binomial_pmf(dp, ao , log_p, log_q) for ao in xrange(min_callable_ao)]
        if not log_pmf_list:
            return 1.0            
              
        log_cdf = min(log_sum_from_log_individual(log_pmf_list), 0.0)
        p_callable = 1.0 - numpy.exp(log_cdf)
                
        if None in [qual_plus, qual_minus] or min_callable_ao == self.__min_var_coverage or (not self.__do_smoothing):
            return p_callable

        # qual_minus should be <= qual_plus
        if qual_minus >= qual_plus:
            return p_callable
            
        # Do linear interpolation
        p_callable_minus = p_callable + numpy.exp(log_pmf_list[-1])
        return linear_interpolation(self.__min_variant_score, qual_minus, p_callable_minus, qual_plus, p_callable)

    # return the 1-st order derivative w.r.t. AF. For Newton's Method used.
    """
    def __callable_prob_and_its_derivative(self, dp, af, min_callable_ao, qual_plus = None, qual_minus = None):
        assert(0.0 <= af <= 1.0)
        log_p = numpy.log(af)
        log_q = BinomialUtils.log_complement_from_log_p(log_p)
        
        ao_ary = numpy.arange(min_callable_ao, dtype = float)
        ro_ary = dp - ao_ary
        log_n_choose_k_ary = numpy.array([BinomialUtils.log_n_choose_k(dp, ao) for ao in xrange(min_callable_ao)], dtype = float)
        
        log_pmf_ary = log_n_choose_k_ary + ao_ary * log_p + ro_ary * log_q
        log_d_positive = log_n_choose_k_ary + ao_ary * log_p + numpy.log(ro_ary) + (ro_ary - 1.0) * log_q
        log_d_negative = log_n_choose_k_ary + numpy.log(ao_ary) + (ao_ary - 1.0) * log_p + ro_ary *log_q
        log_sum_d_positive = log_sum_from_log_individual(log_d_positive)
        log_sum_d_negative = log_sum_from_log_individual(log_d_negative)

        log_abs_derivative = log_abs_diff_from_log(log_sum_d_positive, log_sum_d_negative)
        my_derivative = numpy.exp(log_abs_derivative)
        if log_sum_d_negative > log_sum_d_positive:
            my_derivative *= -1.0
            
        if not log_pmf_ary.size:
            return 1.0, None         

        log_cdf = min(log_sum_from_log_individual(log_pmf_ary), 0.0)
        
        p_callable = 1.0 - numpy.exp(log_cdf)

        # Not yet implemented interpolation
        return p_callable, my_derivative
    """

    # Newton's Method is extremely sensitive to the initial condition.
    # Maybe Gauss-Newton or decent-based method in the future?
    """
    def __calculate_lod_by_newton_method(self, dp):
        min_callable_ao, qual_plus, qual_minus = self.__min_callable_ao(dp)
        # Initial guess of AF
        af = max(0.5* self.__min_allele_freq, 0.5 / dp)

        p_callable_old = -1.0
        af_old = -1.0

        for num_iter in xrange(10):
            p_callable, derivative_at_af = self.__callable_prob_and_its_derivative(dp, af, min_callable_ao, qual_plus, qual_minus)                
            if not (0.0 <= p_callable <= 1.0) or not numpy.isfinite(derivative_at_af):
                return None
            af_old = af
            p_callable_old = p_callable

            delta_af = -(p_callable - self.__min_callable_prob) / derivative_at_af
            af += delta_af
            
            # Stopping rule
            if abs(p_callable_old - p_callable) < 0.001 * self.__min_callable_prob and abs(af_old - af) < 0.001 * self.__min_allele_freq:
                print "num_iter = %d" %num_iter
                return af
                
            if not (0.0 < af < 1.0):
                return None
        return None
    """
        
    def calculate_lod(self, dp):
        return self.__calculate_lod_by_line_search(dp)
        
    # Line search is not too bad due to the monotonicity of P(callable) vs AF
    def __calculate_lod_by_line_search(self, dp):
        # First get the minimum ao that makes the variant callable.
        # Note that min_callable_ao is not a function of AF
        dp = int(dp)
        assert(dp >= 0)
        if dp == 0 or dp < self.__min_var_coverage:
            return None
        min_callable_ao, qual_plus, qual_minus = self.__min_callable_ao(dp)
        if min_callable_ao is None:
            return 1.0
        # Some Setting for the line searching algorithm
        max_rounds = 10
        num_div= 10
        start_af = min(1.0 / (10.0 * dp), 0.1 * self.__min_allele_freq)
        end_af = 1.0 - start_af
        iter_num = 0
        for round_idx in xrange(max_rounds):
            new_start_af = None
            new_end_af = None
            p_callable_new_start = None
            p_callable_new_end = None
            af_ary = numpy.linspace(start_af, end_af, num_div)
            for af in af_ary:
                iter_num += 1
                p_callable = self.__callable_prob(dp, af, min_callable_ao, qual_plus, qual_minus)

                if p_callable < self.__min_callable_prob:
                    new_start_af = af
                    p_callable_new_start = p_callable
                elif p_callable > self.__min_callable_prob:
                    new_end_af = af
                    p_callable_new_end = p_callable
                else:
                    # Lucky me! Exactly hit self.__min_callable_prob.
                    return af
                    
                # Stoping rule in a round
                if None not in [new_start_af, new_end_af]:
                    start_af = new_start_af
                    end_af = new_end_af
                    break
                
            # Stopping if we are very close
            if None not in [p_callable_new_start, p_callable_new_end]:
                if abs(p_callable_new_start - p_callable_new_end) < 0.001 * self.__min_callable_prob:
                    return linear_interpolation(self.__min_callable_prob, p_callable_new_start, new_start_af, p_callable_new_end, new_end_af)

            # Bad initial condition:
            if new_end_af is None:
                end_af = 0.5 * (end_af + 1.0)
            if new_start_af is None:
                start_af = 0.5 * start_af
        
        # Reach the max round, use interpolation.
        if None not in [p_callable_new_start, p_callable_new_end]:
            return linear_interpolation(self.__min_callable_prob, p_callable_new_start, new_start_af, p_callable_new_end, new_end_af)

        return 1.0
# ------------------------------------- End LodManager ---------------------------------------------


if __name__ == '__main__':

    param_dict = {'min_var_coverage': 3, 'min_variant_score': 3, 'min_callable_prob': 0.98, 'min_allele_freq': 0.0005}    
    
    parser = OptionParser("Calculate LOD as a function of Molecular DePth (MPD):")
    parser.add_option('-m', '--mdp',               help='MDP of interests. Comma separated values or use "linspace(start, stop, num)"', dest='mdp')
    parser.add_option('-p', '--parameter-file',    help='Read the TVC hotspot parameters in the files for getting -v, -s, -f ', dest='param_path')    
    parser.add_option('-v', '--min-var-coverage',  help='NOCALL if variant allele molecular coverage below this [%d]'%param_dict['min_var_coverage'], dest='min_var_coverage')    
    parser.add_option('-s', '--min-variant-score', help='NOCALL if variant allele molecular coverage below this [%d]'%param_dict['min_variant_score'], dest='min_variant_score') 
    parser.add_option('-f', '--min-allele-freq',   help='Allele frequency cut-off for a variant call [%s]'%str(param_dict['min_allele_freq']), dest='min_allele_freq') 
    parser.add_option('-c', '--min-callable-prob', help='Minimum callable probability of the variant [%s]'%str(param_dict['min_callable_prob']), dest='min_callable_prob') 
    parser.add_option('-P', '--print-results',     help='Print results in stdout {0, 1} [1]', dest='print_results') 
    parser.add_option('-F', '--fig',               help='(Optional) Path to the output figure', dest='fig_path')
    parser.add_option('-a', '--axis-type',         help='(Optional) X-Y axis for plotting LOD vs MDP. Options: {linear, loglog, semilogx, semilogy} [linear]', dest='axis_type')
    parser.add_option('-j', '--json',              help='(Optional) Path to the output json', dest='json_path')
    (options, args) = parser.parse_args()
    
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)
        
    # mdp
    if options.mdp is None:
        raise(IOError("No mdp provided by -m or --mdp"))
        sys.exit(1)
    elif options.mdp.startswith('linspace(') and options.mdp.endswith(')'):
        dp_list = map(int, eval('numpy.%s' %options.mdp))
    else:
        dp_list = map(int, options.mdp.split(','))
    dp_list.sort()
    if not dp_list:
        raise(IOError("Bad -m or --mdp: Empty list"))
        sys.exit(1)
    if dp_list[0] < 0:
        raise(IOError("MDP (-m, --mdp) can not be a negative value."))
        sys.exit(1)

    # parameter json
    tvc_param_type_dict = {'min_var_coverage': ('hotspot_min_var_coverage', int), 'min_variant_score': ('hotspot_min_variant_score', float), 'min_callable_prob': ('min_callable_prob', float), 'min_allele_freq': ('hotspot_min_allele_freq', float)}
    if options.param_path is not None:
        with open(options.param_path, 'rb') as f_json:
            tvc_param = json.load(f_json)
        for key, type_tuple in tvc_param_type_dict.iteritems():
            param_dict[key] = type_tuple[1](tvc_param.get('torrent_variant_caller', {}).get(type_tuple[0], param_dict[key]))
               
    # min_var_coverage
    if options.min_var_coverage is not None:
        param_dict['min_var_coverage'] = tvc_param_type_dict['min_var_coverage'][1](options.min_var_coverage)

    # min_variant_score
    if options.min_variant_score is not None:
        param_dict['min_variant_score'] = tvc_param_type_dict['min_variant_score'][1](options.min_variant_score)

    # min_callable_prob
    if options.min_callable_prob is not None:
        param_dict['min_callable_prob'] = tvc_param_type_dict['min_callable_prob'][1](options.min_callable_prob)        

    # min_allele_freq
    if options.min_allele_freq is not None:
        param_dict['min_allele_freq'] = tvc_param_type_dict['min_allele_freq'][1](options.min_allele_freq)

    # Calculate LOD    
    lod_manager = LodManager()
    lod_manager.set_parameters(param_dict)
    lod_manager.do_smoothing(True)
    lod_list = [lod_manager.calculate_lod(dp) for dp in dp_list]

    # Print results
    if options.print_results in [None, '1', 'on', 'true', 'True']:
        print '+ LOD vs. MDP:'
        print '  - MDP = [%s]' %(', '.join(map(str, dp_list)))
        print '  - LOD = [%s]' %(', '.join(map(str,lod_list)))
        
    # Dump report
    if options.json_path is not None:
        my_report = {'MDP': dp_list, 'LOD': lod_list, 'parameters': param_dict}
        with open(options.json_path, 'wb') as f_json:
            json.dump(my_report, f_json, indent = 2)
        print '+ Saved json to %s successfully.' %os.path.realpath(options.json_path)
    

    # plot Figure
    if options.fig_path is not None:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt        
        plt.figure(1, figsize=(16,10))
        plt.hold(1)
        if options.axis_type == 'loglog':
            my_plot = plt.loglog
        elif options.axis_type == 'semilogx':
            my_plot = plt.semilogx
        elif options.axis_type == 'semilogy':
            my_plot = plt.semilogy
        else:
            my_plot = plt.plot
        my_plot(dp_list, lod_list, 'b-')
        xmin, xmax = plt.xlim()
        my_plot([xmin, xmax], [param_dict['min_allele_freq']]*2, '--r')
        if options.axis_type not in ['semilogy', 'loglog']:
            ymin, ymax = plt.ylim()
            ymin = min(param_dict['min_allele_freq'] * 0.9, ymin)
            plt.ylim([ymin, ymax])
        plt.legend(['LOD', 'min_allele_freq'])
        plt.grid(1)
        plt.xlabel('MDP')
        plt.ylabel('Frequency')
        title_text = ', '.join('%s=%s' %(k, str(v)) for k, v in param_dict.iteritems()) + '\n'
        plt.title(title_text)
        plt.savefig(options.fig_path)
        print '+ Saved figure to %s successfully.' %os.path.realpath(options.fig_path)