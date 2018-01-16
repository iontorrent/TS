#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (C) 2017 Thermo Fisher Scientific Inc. All Rights Reserved

import copy
import sys

class TvcVcfFile:
    def __init__(self, vcf_path, mode):
        '''
        TvcVcfFile(vcf_path, mode) -> TvcVcfFile object
        Open a vcf file of path vcf_path. 
        Use mode = 'r' (resp. 'w' ) for read (resp. write) a vcf file. 
        '''
        self.__vcf_path = vcf_path
        self.__mode = mode
        if self.__mode not in ['r', 'w']:
            raise(ValueError('mode must be wither \"r\" or \"w\". Not \"%s\"' %mode))
        self.__f_vcf = open(vcf_path, self.__mode)
        if self.__mode == 'r':
            self. __get_header()
            if self.is_missing_header():
                raise(IOError('Missing header information in %s' %self.__vcf_path))
            self.seek(0)
        elif self.__mode == 'w':
            self.__reset_header_info()
            self.__data_to_write = {}
        self.__bypass_size_check_tags = ()
        self.__uniq_flag = False
        self.__allow_same_position = False
        self.__previous = None

    def __enter__(self):
        '''
        __enter__() -> TvcVcfFile object
        Enable with statement
        '''
        return self
    
    def __exit__(self, type, msg, traceback):
        '''
        __exit__() -> bool
        Enable with statement
        '''
        if self.close():
            raise(IOError('Fail to write the vcf records to %s.' %self.__vcf_path))
        return False

    def __flush(self):
        '''
        __flush() -> flush the vcf records stored in self.__data_to_write to the output file (mode = 'w' only)
        '''
        if self.__mode != 'w':
            raise(IOError('File not open for write'))
            return
        if self.is_missing_header():
            raise(ValueError('Can not write to vcf with missing header information'))

        # write the double hash header:            
        self.__f_vcf.write('\n'.join(self.vcf_double_hash_header))
        # write the column header
        self.__f_vcf.write('\n#%s\n' %'\t'.join(self.columns + self.samples))
        for contig_id in self.__contig_id_list:
            data_in_contig = self.__data_to_write[contig_id]
            sorted_pos = map(int, data_in_contig.keys())
            sorted_pos.sort()
            for pos_key in sorted_pos:
                data_to_be_written = set(data_in_contig[str(pos_key)]) if self.__uniq_flag else data_in_contig[str(pos_key)]
                if (not self.__allow_same_position) and len(data_to_be_written) > 1:
                    raise(ValueError('Writing multiple vcf lines at the same position %s:%d is not allowed. Use self.allow_same_position(True) to allow.' %(contig_id, pos_key)))
                for line in data_to_be_written:
                    self.__f_vcf.write('%s\n' %line)
        
    def uniq(self, flag):
        '''
        uniq(flag) -> Bool
        If flag = True remove any records that are exact duplicates when writing the vcf file
        If flag = False do not remove any records that are exact duplicates when writing the vcf file (default)
        '''
        if flag:
            self.__uniq_flag = True
        else:
            self.__uniq_flag = False
        return self.__uniq_flag

    def allow_same_position(self, flag):
        '''
        allow_same_position(flag) -> Bool
        If flag = True, allow different vcf lines at the same chromosome and position
        If flag = False, do not allow different vcf lines at the same chromosome and position (default)
        '''
        if flag:
            self.__allow_same_position = True
        else:
            self.__allow_same_position = False
        return self.__allow_same_position

    def close(self):
        '''
        close() -> int
        If mode = 'w', first self.__flush() and then close the file opened.
        If mode = 'r', simply close the file opened.
        return 0 if success, else return 1. 
        '''
        if self.__mode == 'w':
            try:
                self.__flush()
            except:
                self.__f_vcf.close()
                return 1
        self.__f_vcf.close()
        return 0

    def reset(self):
        '''
        reset() -> Go to the first vcf record (mode = 'r' only).
        '''
        if self.__mode == 'r':
            self.seek(0)
            return
        raise(IOError('File not open for reset.'))
        return

    def seek(self, offset):
        '''
        seek(offset) -> None. Move to the file position.
        Go to the offset of the vcf record where the offset can be obtained by tell() (mode = 'r' only).
        '''
        if self.__mode == 'r':
            if self.__f_vcf.closed:
                self.__f_vcf = open(self.__vcf_path, 'r')
            self.__f_vcf.seek(self.__vcf_record_origin + offset)
            return
        raise(IOError('File not open for seek.'))
        return
        
    def tell(self):
        '''
        tell(offset) -> int 
        Tell the offset of the current vcf record. (mode = 'r' only).
        '''        
        if self.__mode == 'r':
            return self.__f_vcf.tell() - self.__vcf_record_origin
        raise(IOError('File not open for tell.'))
        return        
        
    def __iter__(self):
        '''
        __iter__() -> TvcVcfFile object
        Let TvcVcfFile iteratable.
        '''
        if self.__mode == 'r':
            return self
        raise(IOError('File not open for iterate'))

    def __vcf_record_basic_sanity_check(self, my_vcf_dict, my_offset = None):
        '''
        __vcf_record_basic_sanity_check(dict) -> bool
        Basic sanity check of the vcf record. Return True if pass, False otherwise.
        '''
        # Is CHROM defined in the header?
        try:
            my_contig_idx = self.__contig_id_list.index(my_vcf_dict['CHROM'])
        except ValueError:
            raise(ValueError('The CHROM of the vcf line %s:%d is not in the contigs specified in the header.' %(my_vcf_dict['CHROM'], my_vcf_dict['POS'])))
            return False

        my_pos = my_vcf_dict['POS']

        # Is the reference allele inside the chromosome?
        if my_pos < 1 or my_pos + len(my_vcf_dict['REF']) > self.contig_list[my_contig_idx]['length'] + 1:
            raise(ValueError('The reference allele %s at %s:%d is beyond the contig of length %d.' %(my_vcf_dict['REF'], my_vcf_dict['CHROM'], my_vcf_dict['POS'], self.contig_list[my_contig_idx]['length'])))            
            return False

        # I check sortedness only for the read mode.
        if self.__mode != 'r':
            return True
        
        assert(my_offset is not None) # I must input my_offset
        
        current = {'where': my_offset, 'contig_idx': my_contig_idx, 'pos': my_pos}        
        # I can't check sortedness if I didn't got any record previously.
        if self.__previous is None:
            self.__previous = current
            return True
        
        # Prepare for checking sortedness 
        if current['where'] > self.__previous['where']: # The most common case  (i.e., iterate over the vcf file) comes first
            large_offset = current
            small_offset = self.__previous
        elif current['where'] < self.__previous['where']:
            small_offset = current
            large_offset = self.__previous         
        else:
            assert(current == self.__previous)
            return current == self.__previous

        # Check sortedness
        if large_offset['contig_idx'] == small_offset['contig_idx']: # The most common case: the same chromosome
            if large_offset['pos'] > small_offset['pos']:
                # sorted
                self.__previous = current
                return True
            elif large_offset['pos'] < small_offset['pos']:
                raise(ValueError('The vcf file is not sorted: The line %s:%d comes before the line %s:%d' %(self.contig_list[small_offset['contig_idx']]['ID'], small_offset['pos'], self.contig_list[large_offset['contig_idx']]['ID'], large_offset['pos'])))
            else: # two different lines start at the same position
                if self.__allow_same_position:
                    self.__previous = current
                    return True
                else:
                    raise(ValueError('Two vcf lines are at the same position %s:%d. Use self.allow_same_position(True) to bypass this check.' %(self.contig_list[small_offset['contig_idx']]['ID'], small_offset['pos'])))
            return False
        elif large_offset['contig_idx'] < small_offset['contig_idx']:
            raise(ValueError('The vcf file is not sorted: The line %s:%d comes before the line %s:%d' %(self.contig_list[small_offset['contig_idx']]['ID'], small_offset['pos'], self.contig_list[large_offset['contig_idx']]['ID'], large_offset['pos'])))
            return False
        #else:  # The case of large_offset['contig_idx'] > small_offset['contig_idx']
        self.__previous = current
        return True
    
    def next(self): # Python 3: def __next__(self)
        '''
        next() -> dict. 
        Get the next vcf record in the form of dict (mode = 'r' only).
        '''         
        if self.__mode == 'r':
            line = '#'
            while line.startswith('#'):
                line = self.__f_vcf.readline()
                if line == '':
                    raise(StopIteration)
            
            my_record = self.read_vcf_record(line.strip('\n'))
            my_offset = self.__f_vcf.tell()
            self.__vcf_record_basic_sanity_check(my_record, my_offset)
            return my_record

        raise(IOError('File not open for iterate'))

    def is_missing_header(self):
        '''
        is_missing_header() -> bool 
        Check is there any missing header information (return True if missing header, return False if not)
        ''' 
        missing_header = self.info_field_dict == {} or self.format_tag_dict == {} or self.contig_list == [] or self.__contig_id_list == [] or self.column_to_index == {} or self.columns == [] or self.vcf_double_hash_header == []
        return missing_header
    
    def __reset_header_info(self):
        '''
        __reset_header_info() -> clear all the header information
        '''
        # information retrieve from the header
        self.info_field_dict = {}
        self.format_tag_dict = {}
        self.contig_list = []
        self.__contig_id_list = []
        self.column_to_index = {}
        self.columns = []
        self.sample_to_index = {}
        self.samples = []
        self.vcf_double_hash_header = []
        self.__vcf_record_origin = None
    
    def __get_header(self):
        '''
        __get_header -> get the header informaiton from the opened vcf file (mode = 'r' only)
        '''
        self.__reset_header_info()
        line = None
        while line != '':
            line = self.__f_vcf.readline()        
            line = line.strip('\n')
            if not line:
                continue
            if not line.startswith("#"):
                break
            
            if line.startswith("##"):
                self.__read_header_line(line)
                self.vcf_double_hash_header.append(line.strip('\n'))                                
            else:
                self.__read_column_header(line)
            self.__vcf_record_origin = self.__f_vcf.tell()                

            
    def set_header(self, template_tvc_vcf):
        '''
        set_header(template_tvc_vcf) -> set the header from the TvcVcfFile object
        Copy all the header and sample information from template_tvc_vcf to self.
        '''
        if self.__mode == 'w':
            if template_tvc_vcf.is_missing_header():
                raise(ValueError('Can not set header from the template_tvc_vcf. The header information in template_tvc_vcf is missing.'))
                return
            if self.__data_to_write != {}:
                print('WARNING: set_header: All the previously written vcf records will be lost.')
                self.__data_to_write = {}
            
            self.info_field_dict = copy.deepcopy(template_tvc_vcf.info_field_dict)
            self.format_tag_dict = copy.deepcopy(template_tvc_vcf.format_tag_dict)
            self.contig_list = copy.deepcopy(template_tvc_vcf.contig_list)
            self.column_to_index = copy.deepcopy(template_tvc_vcf.column_to_index)
            self.columns = copy.deepcopy(template_tvc_vcf.columns)
            self.vcf_double_hash_header = copy.deepcopy(template_tvc_vcf.vcf_double_hash_header)
            self.set_samples(template_tvc_vcf.samples)
            # sanity check
            for column, index in self.column_to_index.iteritems():
                assert(self.columns[index] == column)
            self.__contig_id_list = [contig['ID'] for contig in self.contig_list]
            self.__data_to_write = dict([(contig_id, {}) for contig_id in self.__contig_id_list])
            return
        else:
            raise(IOError('File open not for changing header.'))
            return
        
    def set_samples(self, sample_list):
        '''
        set_samples(sample_list) -> set the sample information from sample_list.
        '''        
        if self.__mode == 'w':
            if self.__data_to_write != {}:
                print('WARNING: set_samples: All the previously written vcf records will be lost.')
                self.__data_to_write = {}
            
            self.samples = copy.deepcopy(sample_list)
            self.sample_to_index = dict([(key, index) for index, key in enumerate(self.samples)])
            return
        else:
            raise(IOError('File open not for changing samples.'))
            return        

    def __read_column_header(self, line):
        if line.startswith('##'):
            return
        if not line.startswith('#CHROM'):
            return
        splitted_line = line.strip('#').split('\t')
        format_idx = splitted_line.index('FORMAT')
        self.column_to_index = dict([(key, index) for index, key in enumerate(splitted_line[:format_idx + 1])])
        self.columns = [v for v in splitted_line[:format_idx + 1]]
        self.samples = [value for value in splitted_line[format_idx + 1:]]
        self.sample_to_index = dict([(key, index) for index, key in enumerate(self.samples)])

    def __read_info_and_format_from_header(self, line):
        find_text_list = ['ID=', ',Number=', ',Type=', ',Description=']
        find_indices = [line.index(find_text) for find_text in find_text_list]
        my_dict = {'ID': line[find_indices[0] + len(find_text_list[0]) : find_indices[1]],
                   'Number': line[find_indices[1] + len(find_text_list[1]) : find_indices[2]],
                   'Type': line[find_indices[2] + len(find_text_list[2]) : find_indices[3]],
                   'Description': line[find_indices[3] + len(find_text_list[3]) : -1]}
        # Use my_dict['TypeConvMethod'] to do data conversion
        my_dict['TypeConvMethod'] = {'Float': float, 'Integer': int, 'String': str}.get(my_dict['Type'], lambda x : None)
        my_dict['IsMultipleEntries'] = self.__is_multiple_entries(my_dict)
        try:
            my_dict['Number'] = int(my_dict['Number'])
        except ValueError:
            pass        
        return my_dict

    def __read_contig_from_header(self, line):
        find_text_list = ['ID=', ',length=', ',assembly=']
        find_indices = [line.index(find_text) for find_text in find_text_list]
        my_dict = {'ID': line[find_indices[0] + len(find_text_list[0]) : find_indices[1]],
                   'length': int(line[find_indices[1] + len(find_text_list[1]) : find_indices[2]]),
                   'assembly': line[find_indices[2] + len(find_text_list[2]) : -1]}
        self.contig_list.append(my_dict)
        if my_dict['ID'] in self.__contig_id_list:
            raise(ValueError('The header has duplicated contig id: %s' %my_dict['ID']))
        self.__contig_id_list.append(my_dict['ID'])
        

    def __read_header_line(self, line):
        if line.startswith("##FORMAT=<") and line.endswith('>'):
            my_dict = self.__read_info_and_format_from_header(line)
            self.format_tag_dict[my_dict['ID']] = my_dict
            return
        elif line.startswith("##INFO=<") and line.endswith('>'):
            my_dict = self.__read_info_and_format_from_header(line)
            self.info_field_dict[my_dict['ID']] = my_dict
            return
        elif line.startswith("##contig=<") and line.endswith('>'):
            self.__read_contig_from_header(line)
            return
        else:
            return

    def __convert_to_type(self, text, my_dict):
        if text == '.':
            return text
        value = my_dict['TypeConvMethod'](text)
        if value is None:
            if my_dict['Type'] == 'Flag':
                raise(ValueError('A \"Flag\" can not have a value.'))
            else:
                raise(ValueError('Unsupported type for data conversion: %s' %my_dict['Type']))
        return value
    
    def set_bypass_size_check_tags(self, tags):
        '''
        set_bypass_size_check_tags(tags) -> set the tags in INFO FILED or FORMAT that will bypass the size check.
        tags can be a string or list of string. 
        '''
        if type(tags) is str:
            self.__bypass_size_check_tags = tuple([tags, ])
        else:
            self.__bypass_size_check_tags  = tuple(tags)

    def __decode_value(self, my_value, my_dict, num_alt):
        if my_dict['IsMultipleEntries']:
            splitted_value = my_value.split(',')
            self.__check_size(splitted_value, my_dict, num_alt)            
            return [self.__convert_to_type(value, my_dict) for value in splitted_value]   
        else:
            return self.__convert_to_type(my_value, my_dict)
        
    def __encode_one_value(self, value):
        # The convention is to set 0.0 and 1.0 to be '0' and '1' in the vcf record
        return str(int(value)) if value in [0.0, 1.0] else str(value)
    
    def __encode_values(self, my_value, my_dict, num_alt):
        self.__check_size(my_value, my_dict, num_alt)
        if my_dict['IsMultipleEntries']:
            return ','.join([self.__encode_one_value(self.__convert_to_type(v, my_dict)) for v in my_value])
        else:
            return self.__encode_one_value(self.__convert_to_type(my_value, my_dict))

    def __get_key_dict(self, key, is_info_field):
        '''
        is_info_field = True if it is info field, False if it is format tag 
        '''
        try:
            return self.info_field_dict[key] if is_info_field else self.format_tag_dict[key]
        except KeyError:
            raise(KeyError('Unknow key \"%s\" in %s: not specified in the header.' %(key, 'INFO' if is_info_field else 'FORMAT')))
            return None
            
    def __is_multiple_entries(self, format_dict):
        if format_dict['Number'] in [0, 1]:
            return False
        if format_dict['Number'] in ['A', '.']:
            return True
        if type(format_dict['Number']) is int and format_dict['Number'] > 1:
            return True
        return False
            
    def __check_size(self, value, format_dict, num_alt):
        if value in ['.', ['.',]] or format_dict['Number'] == '.' or format_dict['ID'] in self.__bypass_size_check_tags:
            # TS-14886, IR-29193
            return True

        my_len = len(value) if type(value) in [list, tuple] else 1
        expected_len = format_dict['Number'] if format_dict['Number'] != 'A' else num_alt
        if expected_len != my_len:
            if format_dict['Number'] == 'A':
                raise(IndexError('The length of %s does not match len(ALT) = %d' %(str({format_dict['ID']: value}), num_alt)))
            else:
                raise(TypeError('%s conflicts the \"Number\" specified in the header.'%str({format_dict['ID']: value})))
            return False
        return True

    
    def __decode_format_tag_one_entry(self, tag_key, tag_value, num_alt):
        my_tag_dict = self.__get_key_dict(tag_key, False)
        return tag_key, self.__decode_value(tag_value, my_tag_dict, num_alt)

    def __decode_format_tag(self, format_keys, samples_tags, num_alt):
        splitted_format_tag_key = format_keys.split(':')
        format_tag_list = []
        for sample_idx in xrange(len(self.samples)):
            splitted_format_tag_value = samples_tags[sample_idx].split(':')
            if len(splitted_format_tag_key) != len(splitted_format_tag_value):
                raise(ValueError('FORMAT \"%s\"and TAGS \"%s\"mismatches:'%(format_keys, samples_tags[sample_idx])))
                return {}
            format_tag_list.append(dict([self.__decode_format_tag_one_entry(tag_key, splitted_format_tag_value[tag_idx], num_alt) for tag_idx, tag_key in enumerate(splitted_format_tag_key)]),)
        return format_tag_list

    def __decode_info_field(self, info_text, num_alt):
        zipped_info_fields = [self.__decode_info_field_one_entry(info_text_one_entry, num_alt) for info_text_one_entry in info_text.split(';')]
        return dict(zipped_info_fields)

    def __decode_info_field_one_entry(self, info_text_one_entry, num_alt):
        if '=' not in info_text_one_entry:
            info_key = info_text_one_entry
            info_value = None
        else:
            splitted_info_text = info_text_one_entry.split('=')
            if len(splitted_info_text) > 2:
                raise(ValueError('The INFO FIELD \"%s\" contains more that one \"=\"' %info_text_one_entry))
                return None, None
            info_key = splitted_info_text[0]
            info_value = splitted_info_text[1]

        my_info_dict = self.__get_key_dict(info_key, True)

        my_type = my_info_dict['Type']
        if info_value is None: 
            if my_type != 'Flag':
                raise(TypeError('The INFO FIELD \"%s\" conflicts %s' %(info_text_one_entry, str(my_info_dict))))
            return info_key, info_value
        
        return info_key, self.__decode_value(info_value, my_info_dict, num_alt)


    def read_vcf_record(self, line):
        '''
        read_vcf_record(line) -> dict
        Decode a text of vcf record to a dict
        '''
        if ' ' in line:
            raise(ValueError('A valid vcf record can not have a space: \"%s\"'%line))
        splitted_line = line.strip('\n').split('\t')
        alt_list = splitted_line[self.column_to_index['ALT']].split(',')
        num_alt = len(alt_list)
        try:
            vcf_dict = {'CHROM': splitted_line[self.column_to_index['CHROM']], 
                        'POS': int(splitted_line[self.column_to_index['POS']]),
                        'ID': splitted_line[self.column_to_index['ID']],
                        'REF': splitted_line[self.column_to_index['REF']],
                        'ALT': alt_list,
                        'QUAL': float(splitted_line[self.column_to_index['QUAL']]),
                        'FILTER': splitted_line[self.column_to_index['FILTER']],
                        'INFO': self.__decode_info_field(splitted_line[self.column_to_index['INFO']], num_alt),
                        'FORMAT': self.__decode_format_tag(splitted_line[self.column_to_index['FORMAT']], splitted_line[self.column_to_index['FORMAT'] + 1 :], num_alt),
                        'FORMAT_ORDER': splitted_line[self.column_to_index['FORMAT']].split(':'),
                        }
        except Exception as e:
            print 'Error in vcf line: ' + line
            raise type(e), type(e)(e.message +
                               ' happens at %s' % line), sys.exc_info()[2]

        return vcf_dict

    def __encode_info_field(self, info_dict, num_alt):
        info_list = []
        info_keys = info_dict.keys()
        info_keys.sort()
        for key in info_keys:
            my_info_dict = self.__get_key_dict(key, True)
            if my_info_dict['Type'] == 'Flag':
                if key == 'HS' :
                    # The convention is to put HS in the end of INFO FIELD
                    continue 
                info_list.append(key)
            else:
                info_list.append('%s=%s'%(key, self.__encode_values(info_dict[key], my_info_dict, num_alt)))
        if 'HS' in info_dict and self.info_field_dict['HS']['Type'] == 'Flag':
            # The convention is to put HS in the end of INFO FIELD
            info_list.append('HS')
        return ';'.join(info_list)

    def __encode_format_tag(self, format_tag_dict, num_alt, format_order = None):
        if format_order is None:
            format_order = format_tag_dict.keys()
            format_order.sort()
        else:
            if len(format_order) != len(format_tag_dict.keys()):
                raise(IndexError('Order of the format tags does not match the tags.'))

        format_tag_list = []
        for key in format_order:
            my_format_dict = self.__get_key_dict(key, False)
            format_tag_list.append(self.__encode_values(format_tag_dict[key], my_format_dict, num_alt))

        return ':'.join(format_order), format_order,':'.join(format_tag_list)

    def write(self, vcf_dict):
        '''
        write(vcf_dict) -> write the vcf record to the buffer (mode = 'w' only)
        Encode vcf_dict to the text format and save it to the buffer.
        The vcf record will not be written to the file until self.__flush()
        '''
        if self.__mode != 'w':
            raise(IOError('File not open for write.'))
            return
        
        vcf_text = self.vcf_dict_to_text(vcf_dict)
        my_lines_at_pos = self.__data_to_write[vcf_dict['CHROM']].setdefault(str(vcf_dict['POS']), [])
        my_lines_at_pos.append(vcf_text)

    def vcf_dict_to_text(self, vcf_dict):
        '''
        vcf_dict_to_text(vcf_dict) -> str
        Encode vcf_dict to the text format.
        '''        
        alt = ','.join(vcf_dict['ALT'])
        num_alt = len(vcf_dict['ALT'])
        info_field_text = self.__encode_info_field(vcf_dict['INFO'], num_alt)
        if len(vcf_dict['FORMAT']) != len(self.samples):
            raise(IndexError('FORMAT TAG does not match the number of smaples'))
            return None
        format_order = vcf_dict.get('FORMAT_ORDER', None)
        format_tag_list = []
        for format_tag_dict in vcf_dict['FORMAT']:
            my_format_order_text, my_format_order, format_tag = self.__encode_format_tag(format_tag_dict, num_alt, format_order)
            if format_order is None:
                format_order = my_format_order
            elif format_order != my_format_order:
                raise(ValueError('The samples have different FORMAT.'))
            format_tag_list.append(format_tag)
        my_vcf_text_dict = {'CHROM': vcf_dict['CHROM'], 
                    'POS': str(vcf_dict['POS']),
                    'ID': vcf_dict['ID'],
                    'REF': vcf_dict['REF'],
                    'ALT': alt,
                    'QUAL': str(vcf_dict['QUAL']),
                    'FILTER': vcf_dict['FILTER'],
                    'INFO': info_field_text, 
                    'FORMAT': my_format_order_text, }
        splitted_vcf_text = [None, ] * (self.column_to_index['FORMAT'] + 1)
        for key, index in self.column_to_index.iteritems():
            splitted_vcf_text[index] = my_vcf_text_dict[key]
        splitted_vcf_text += format_tag_list

        vcf_line = '\t'.join(splitted_vcf_text)
        return vcf_line
        

    def add_to_header(self, new_header_text):
        '''
        add_to_header(new_header_txt)
        Add a new header entry to the vcf
        '''
        self.__read_header_line(new_header_text)
        self.vcf_double_hash_header.append(new_header_text)
        
if __name__ == '__main__':
    """
    # Example 1: read the vcf line by line.
    vcf_path = 'small_variants.vcf'
    record_idx = 0
    idx_of_interests = 100
    all_vcf_records = []    
    with TvcVcfFile(vcf_path, 'r') as f_vcf:
        # TVC abuses the FR tag, which is illigal actually. I want to bypass the size check for FR
        f_vcf.set_bypass_size_check_tags(['FR',])
        for vcf_dict in f_vcf:
            # Method: convert the vcf dict to text
            text_line = f_vcf.vcf_dict_to_text(vcf_dict)
            # Method: convert the text line to vcf_dict
            vcf_dict_2 = f_vcf.read_vcf_record(text_line)
            # Method: tell me where to find vcf_dict_of_interests            
            if record_idx == idx_of_interests - 1:
                file_offset = f_vcf.tell()
            all_vcf_records.append(vcf_dict)
            record_idx += 1
        # Method: use seek to find vcf_dict_of_interests
        f_vcf.seek(file_offset)
        print(all_vcf_records[idx_of_interests] == f_vcf.next())
            
    # Example 2: edit and write the vcf file
    vcf_r_path = 'small_variants.vcf'
    vcf_w_path = 'small_variants_filtered.vcf'
    f_vcf_r = TvcVcfFile(vcf_r_path, 'r')
    f_vcf_w = TvcVcfFile(vcf_w_path, 'w')
    # f_vcf_w uses the headers and samples of f_vcf_r
    f_vcf_w.set_header(f_vcf_r)
    for vcf_dict in f_vcf_r:
        # Edit the vcf record
        vcf_dict['FILTER'] = 'NOCALL'
        f_vcf_w.write(vcf_dict)
    f_vcf_r.close()
    f_vcf_w.close()
    

    # Example 3: write the vcf file qith modified sample
    vcf_r_path = 'small_variants.vcf'
    vcf_w_path = 'small_variants_split_sample.vcf'
    f_vcf_r = TvcVcfFile(vcf_r_path, 'r')
    f_vcf_w = TvcVcfFile(vcf_w_path, 'w')
    # f_vcf_w uses the headers of f_vcf_r
    f_vcf_w.set_header(f_vcf_r)
    # Set samples of interests for write
    sample_idx_of_interests = [1, 2]
    samples = [f_vcf_r.samples[idx] for idx in sample_idx_of_interests]
    f_vcf_w.set_samples(samples)
    for vcf_dict in f_vcf_r:
        # Output the FORMAT TAGS for samples specified in sample_idx_of_interests only
        vcf_dict['FORMAT'] = [vcf_dict['FORMAT'][idx] for idx in sample_idx_of_interests]
        f_vcf_w.write(vcf_dict)
    f_vcf_r.close()
    f_vcf_w.close()    
    """
    pass