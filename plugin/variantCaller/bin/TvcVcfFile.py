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
        for contig in self.contig_list:
            data_in_contig = self.__data_to_write[contig['ID']]
            sorted_pos = [int(p) for p in data_in_contig.keys()]
            sorted_pos.sort()
            for pos_key in sorted_pos:
                if self.__uniq_flag:
                    for line in set(data_in_contig[str(pos_key)]):
                        self.__f_vcf.write('%s\n' %line)
                else:
                    for line in data_in_contig[str(pos_key)]:
                        self.__f_vcf.write('%s\n' %line)
        
    def uniq(self, flag):
        '''
        uniq(flag) -> Bool
        If flag = True remove any records that are exact duplicates
        If flag = False do not remove any records that are exact duplicates (default)
        '''
        if flag:
            self.__uniq_flag = True
        else:
            self.__uniq_flag = False
        return self.__uniq_flag

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
            return self.read_vcf_record(line.strip('\n'))

        raise(IOError('File not open for iterate'))

    def is_missing_header(self):
        '''
        is_missing_header() -> bool 
        Check is there any missing header information (return True if missing header, return False if not)
        ''' 
        missing_header = self.info_field_dict == {} or self.format_tag_dict == {} or self.contig_list == [] or self.column_to_index == {} or self.columns == [] or self.vcf_double_hash_header == []
        return missing_header
    
    def __reset_header_info(self):
        '''
        __reset_header_info() -> clear all the header information
        '''
        # information retrieve from the header
        self.info_field_dict = {}
        self.format_tag_dict = {}
        self.contig_list = []
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
            for column, index in self.column_to_index .iteritems():
                assert(self.columns[index] == column)    
            self.__data_to_write = dict([(contig['ID'], {}) for contig in self.contig_list])
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

    def __convert_to_type(self, text, text_type):
        if text == '.':
            return text
        if text_type == 'Integer':
            return int(text)
        if text_type == 'Float':
            return float(text)
        if text_type == 'String':
            return str(text)
        if text_type == 'Flag':
            raise(ValueError('A \"Flag\" should not have a value.'))
        raise(ValueError('Unknown type: %s' %text_type))
        return None
    
    def set_bypass_size_check_tags(self, tags):
        '''
        set_bypass_size_check_tags(tags) -> set the tags in INFO FILED or FORMAT that will bypass the size check.
        tags can be a string or list of string. 
        '''
        if type(tags) is str:
            self.__bypass_size_check_tags = tuple([tags, ])
        else:
             self.__bypass_size_check_tags  = tuple(tags)

    def __decode_value(self, my_key, my_value, my_dict, num_alt):
        my_type = my_dict['Type']
        if self.__is_multiple_entries(my_dict):
            splitted_value = my_value.split(',')
            self.__check_size(my_key, splitted_value, my_dict, num_alt)            
            return [self.__convert_to_type(value, my_type) for value in splitted_value]   
        else:
            return self.__convert_to_type(my_value, my_type)  

    def __encode_values(self, my_key, my_value, my_dict, num_alt):
        my_type = my_dict['Type']
        self.__check_size(my_key, my_value, my_dict, num_alt)
        if self.__is_multiple_entries(my_dict):
            return ','.join([str(self.__convert_to_type(v, my_type)) for v in my_value])
        else:
            return str(self.__convert_to_type(my_value, my_type))

    def __get_key_dict(self, key, source_of_key):
        if source_of_key == 'info field':
            header_dict = self.info_field_dict            
        elif source_of_key == 'format tag':
            header_dict = self.format_tag_dict            
        else:
            raise(KeyError('Unkown source of key: %s' %source_of_key))
            return None
        try:
            return header_dict[key]
        except KeyError:
            raise(KeyError('Unknow key \"%s\" in the %s: not specified in the header.' %(key, source_of_key)))
            return None
            
    def __is_multiple_entries(self, format_dict):
        if format_dict['Number'] in [0, 1]:
            return False
        if format_dict['Number'] in ['A', '.']:
            return True
        if type(format_dict['Number']) is int and format_dict['Number'] > 1:
            return True
        return False
            
    def __check_size(self, key, value, format_dict, num_alt):
        if key in self.__bypass_size_check_tags:
            return
        try:
            # TS-14886, IR-29193
            if list(value) == ['.', ]:
                return
        except TypeError:
            pass
        if format_dict['Number'] in [0, 1]:
            if type(value) is list or type(value) is tuple:
                raise(TypeError('%s conflicts the \"Number\" specified in the header.'%str({key, value})))
        if format_dict['Number'] == 'A':
            if len(value) != num_alt:
                raise(IndexError('The length of %s does not match the number of ALT = %d' %(str({key: value}), num_alt)))
        elif type(format_dict['Number']) is int and format_dict['Number'] > 1:
            if (len(value) != format_dict['Number']):
                raise(IndexError('The length of %s does not match the one specified in the header %d' %(str({key: value}, format_dict['Number']))))
    
    
    def __decode_format_tag_one_entry(self, tag_key, tag_value, num_alt):
        my_tag_dict = self.__get_key_dict(tag_key, 'format tag')
        return tag_key, self.__decode_value(tag_key, tag_value, my_tag_dict, num_alt)

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

        my_info_dict = self.__get_key_dict(info_key, 'info field')

        my_type = my_info_dict['Type']
        if info_value is None: 
            if my_type != 'Flag':
                raise(TypeError('The INFO FIELD \"%s\" conflicts %s' %(info_text_one_entry, str(my_info_dict))))
            return info_key, info_value
        
        return info_key, self.__decode_value(info_key, info_value, my_info_dict, num_alt)


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
            my_info_dict = self.__get_key_dict(key, 'info field')
            if my_info_dict['Type'] == 'Flag':
                if key == 'HS' :
                    # The convention is to put HS in the end of INFO FIELD
                    continue 
                info_list.append(key)
            else:
                info_list.append('%s=%s'%(key, self.__encode_values(key, info_dict[key], my_info_dict, num_alt)))
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
            my_format_dict = self.__get_key_dict(key, 'format tag')
           
            format_tag_list.append('%s'%(self.__encode_values(key, format_tag_dict[key], my_format_dict, num_alt)))

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
        pos_key = str(vcf_dict['POS'])
        if pos_key not in self. __data_to_write[vcf_dict['CHROM']]:
            self.__data_to_write[vcf_dict['CHROM']][pos_key] = [vcf_text, ]
        else:
            self.__data_to_write[vcf_dict['CHROM']][pos_key].append(vcf_text)

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
