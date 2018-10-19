#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (C) 2017 Thermo Fisher Scientific Inc. All Rights Reserved

import copy
from functools import wraps

TVC_FILE_MODE_ATTR = '_TvcVcfFile__mode'


def read_mode_only(func):
    """
    raise IOError when mode is not 'r'
    :param func:
    :return:
    """
    @wraps(func)
    def method_wrapper(self, *args, **kwargs):
        if not hasattr(self, TVC_FILE_MODE_ATTR):
            msg = 'Required attribute (__mode) is not found.'
            raise AttributeError(msg)

        if getattr(self, TVC_FILE_MODE_ATTR) is not 'r':
            msg = 'File not open in read (r) mode for %s' % func.__name__
            raise IOError(msg)
        return func(self, *args, **kwargs)
    return method_wrapper


def write_mode_only(func):
    """
    raise IOError when mode is not 'w'
    :param func:
    :return:
    """
    @wraps(func)
    def method_wrapper(self, *args, **kwargs):
        if not hasattr(self, TVC_FILE_MODE_ATTR):
            msg = 'Required attribute (__mode) is not found.'
            raise AttributeError(msg)

        if getattr(self, TVC_FILE_MODE_ATTR) is not 'w':
            msg = 'File not open in write (w) mode for %s' % func.__name__
            raise IOError(msg)
        return func(self, *args, **kwargs)
    return method_wrapper


class TvcVcfFile:
    def __init__(self, vcf_path, mode):
        """
        TvcVcfFile(vcf_path, mode) -> TvcVcfFile object
        Open a vcf file of path vcf_path.
        Use mode = 'r' (resp. 'w' ) for read (resp. write) a vcf file.
        """
        self.__vcf_path = vcf_path
        self.__mode = mode
        if self.__mode not in ['r', 'w']:
            raise ValueError('mode must be wither \"r\" or \"w\". Not \"%s\"' %mode)
        self.__f_vcf = open(vcf_path, self.__mode)
        if self.__mode == 'r':
            self. __get_header()
            if self.is_missing_header():
                raise IOError('Missing header information in %s' %self.__vcf_path)
            self.seek(0)
        elif self.__mode == 'w':
            self.__reset_header_info()
            self.__data_to_write = {}
            self.__num_records_in_write_buffer = 0
            self.__is_header_written = False
        self.__bypass_size_check_tags = ()
        self.__uniq_flag = False
        self.__allow_same_position = False
        # if open for read, self.__previous is the last record I read.
        # if open for write, self.__previous is the last record I flushed.
        self.__previous = None


    def __enter__(self):
        """
        __enter__() -> TvcVcfFile object
        Enable with statement
        """
        return self

    def __exit__(self, type, msg, traceback):
        """
        __exit__() -> bool
        Enable with statement
        """
        if self.close():
            raise IOError('Fail to write the vcf records to %s.' %self.__vcf_path)
        return False

    def get_num_records_in_write_buffer(self):
        return self.__num_records_in_write_buffer

    def get_last_position_in_write_buffer(self):
        if self.__mode == 'r':
            return None

        if self.__num_records_in_write_buffer == 0:
            return None

        for contig_id in self.__contig_id_list[::-1]:
            keys = list(self.__data_to_write[contig_id].keys())
            if keys:
                return {'contig': contig_id, 'pos': max(keys)}

        return None

    @write_mode_only
    def flush(self):
        """
        flush() -> flush the vcf records stored in self.__data_to_write to the output file (mode = 'w' only)
        Note that once you call this function, you can not write a record comes before the record that has been flushed.
        If you are not sure about it, please do not call this function.
        Instead, the flush will be done automatically in self.close().
        """
        if self.is_missing_header():
            raise ValueError('Can not write to vcf with missing header information')

        # Flush the header if I haven't done yet.
        if not self.__is_header_written:
            # write the double hash header:
            self.__f_vcf.write('\n'.join(self.vcf_double_hash_header))
            # write the column header
            self.__f_vcf.write('\n#%s\n' %'\t'.join(self.columns + self.samples))
            # Header has been written
            self.__is_header_written = True

        # No record to flush
        if self.__num_records_in_write_buffer == 0:
            return

        # Write the record
        for contig_idx, contig_id in enumerate(self.__contig_id_list):
            data_in_contig = self.__data_to_write[contig_id]
            sorted_pos = list(data_in_contig.keys())
            # Sort the position in the contig
            sorted_pos.sort()
            for pos_key in sorted_pos:
                data_to_be_written = set(data_in_contig[pos_key]) if self.__uniq_flag else data_in_contig[pos_key]
                if (not self.__allow_same_position) and len(data_to_be_written) > 1:
                    raise ValueError('Writing multiple vcf lines at the same position %s:%d is not allowed. Use self.allow_same_position(True) to allow.' %(contig_id, pos_key))
                for line in data_to_be_written:
                    self.__f_vcf.write('%s\n' %line)
                    self.__previous = {'contig_idx': contig_idx, 'pos': pos_key}
            # Clear all data in the buffer.
            data_in_contig.clear()

        # Reset __num_records_in_write_buffer
        self.__num_records_in_write_buffer = 0

    def uniq(self, flag):
        """
        uniq(flag) -> Bool
        If flag = True remove any records that are exact duplicates when writing the vcf file
        If flag = False do not remove any records that are exact duplicates when writing the vcf file (default)
        """
        if flag:
            self.__uniq_flag = True
        else:
            self.__uniq_flag = False
        return self.__uniq_flag

    def allow_same_position(self, flag):
        """
        allow_same_position(flag) -> Bool
        If flag = True, allow different vcf lines at the same chromosome and position
        If flag = False, do not allow different vcf lines at the same chromosome and position (default)
        """
        if flag:
            self.__allow_same_position = True
        else:
            self.__allow_same_position = False
        return self.__allow_same_position

    def close(self):
        """
        close() -> int
        If mode = 'w', first self.flush() and then close the file opened.
        If mode = 'r', simply close the file opened.
        return 0 if success, else return 1.
        """
        if self.__mode == 'w':
            try:
                self.flush()
            except:
                self.__f_vcf.close()
                return 1
        self.__f_vcf.close()
        return 0

    @read_mode_only
    def reset(self):
        """
        reset() -> Go to the first vcf record (mode = 'r' only).
        """
        self.seek(0)
        return

    @read_mode_only
    def seek(self, offset):
        """
        seek(offset) -> None. Move to the file position.
        Go to the offset of the vcf record where the offset can be obtained by tell() (mode = 'r' only).
        """
        if self.__f_vcf.closed:
            self.__f_vcf = open(self.__vcf_path, 'r')
        self.__f_vcf.seek(self.__vcf_record_origin + offset)
        return

    @read_mode_only
    def tell(self):
        """
        tell(offset) -> int
        Tell the offset of the current vcf record. (mode = 'r' only).
        """
        return self.__f_vcf.tell() - self.__vcf_record_origin

    @read_mode_only
    def __iter__(self):
        """
        __iter__() -> TvcVcfFile object
        Let TvcVcfFile iteratable.
        """
        return self

    def is_valid_position(self, allele_dict):
        '''
        __is_valid_position(dict) -> bool
        Basic sanity check of the contig and position defined in allele_dict. Return True if valid, False otherwise.
        allele_dict = {'pos': position, 'contig': contig, 'contig_idx': contig_idx} where 'contig' or 'contig_idx' must be presented.
        '''
        my_contig_idx = allele_dict.get('contig_idx', None)
        if my_contig_idx is None:
            my_contig_idx = self.__contig_id_list.index[allele_dict['contig']]
        my_pos = int(allele_dict['pos'])
        my_ref = allele_dict.get('ref', '\0')
        if my_pos < 1 or my_pos + len(my_ref) > self.contig_list[my_contig_idx]['length'] + 1:
            return False
        return True

    def compare_positions(self, allele_dict_0, allele_dict_1):
        '''
        compare_positions(dict, dict) -> int
        Compare the two positions defined by allele_dict_0 and allele_dict_1.
        return 0 if they are at the same position, 1 if allele_dict_0 comes before allele_dict_1, -1 if allele_dict_0 comes after allele_dict_1.
        '''
        contig_idx_0 = allele_dict_0.get('contig_idx', None)
        if contig_idx_0 is None:
            contig_idx_0 = self.__contig_id_list.index(allele_dict_0.get('contig', allele_dict_0.get('CHROM', None)))
        contig_idx_1 = allele_dict_1.get('contig_idx', None)
        if contig_idx_1 is None:
            contig_idx_1 = self.__contig_id_list.index(allele_dict_1.get('contig', allele_dict_1.get('CHROM', None)))
        if contig_idx_0 < contig_idx_1:
            return 1
        elif contig_idx_0 > contig_idx_1:
            return -1
        pos_0 = int(allele_dict_0.get('pos', allele_dict_0.get('POS', None)))
        pos_1 = int(allele_dict_1.get('pos', allele_dict_1.get('POS', None)))
        if  pos_0 < pos_1:
            return 1
        elif pos_0 > pos_1:
            return -1
        return 0

    def __vcf_record_basic_sanity_check(self, my_vcf_dict, my_offset = None):
        """
        __vcf_record_basic_sanity_check(dict) -> bool
        Basic sanity check of the vcf record. Return True if pass, False otherwise.
        """
        # Is CHROM defined in the header?
        try:
            my_contig_idx = self.__contig_id_list.index(my_vcf_dict['CHROM'])
        except ValueError:
            raise ValueError('The CHROM of the vcf line %s:%d is not in the contigs specified in the header.' %(my_vcf_dict['CHROM'], my_vcf_dict['POS']))
            return False

        my_pos = my_vcf_dict['POS']

        # Is the reference allele inside the chromosome?
        if not self.is_valid_position({'contig_idx': my_contig_idx, 'pos': my_pos, 'ref': my_vcf_dict['REF']}):
            raise ValueError('The reference allele %s at %s:%d is beyond the contig of length %d.' %(my_vcf_dict['REF'], my_vcf_dict['CHROM'], my_vcf_dict['POS'], self.contig_list[my_contig_idx]['length']))
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
        compare_positions = self.compare_positions(small_offset, large_offset)
        if compare_positions > 0:
            self.__previous = current
            return True
        elif compare_positions < 0:
            raise ValueError('The vcf file is not sorted: The line %s:%d comes before the line %s:%d' %(self.contig_list[small_offset['contig_idx']]['ID'], small_offset['pos'], self.contig_list[large_offset['contig_idx']]['ID'], large_offset['pos']))
        else:
            if self.__allow_same_position:
                self.__previous = current
                return True
            else:
                raise ValueError('Two vcf lines are at the same position %s:%d. Use self.allow_same_position(True) to bypass this check.' %(self.contig_list[small_offset['contig_idx']]['ID'], small_offset['pos']))
        self.__previous = current
        return True

    @read_mode_only
    def __next__(self):
        """
        next() -> dict.
        Get the next vcf record in the form of dict (mode = 'r' only).
        """
        line = '#'
        while line.startswith('#'):
            line = self.__f_vcf.readline()
            if line == '':
                raise StopIteration

        my_record = self.read_vcf_record(line.strip('\n'))
        my_offset = self.__f_vcf.tell()
        self.__vcf_record_basic_sanity_check(my_record, my_offset)
        return my_record

    # python 2 compatibility
    next = __next__

    def is_missing_header(self):
        """
        is_missing_header() -> bool
        Check is there any missing header information (return True if missing header, return False if not)
        """
        missing_header = self.info_field_dict == {} or self.format_tag_dict == {} or self.contig_list == [] or self.__contig_id_list == [] or self.column_to_index == {} or self.columns == [] or self.vcf_double_hash_header == []
        return missing_header

    def __reset_header_info(self):
        """
        __reset_header_info() -> clear all the header information
        """
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
        """
        __get_header -> get the header informaiton from the opened vcf file (mode = 'r' only)
        """
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

    @write_mode_only
    def set_header(self, template_tvc_vcf):
        """
        set_header(template_tvc_vcf) -> set the header from the TvcVcfFile object
        Copy all the header and sample information from template_tvc_vcf to self.
        """
        if template_tvc_vcf.is_missing_header():
            raise ValueError(
                'Can not set header from the template_tvc_vcf. The header information in template_tvc_vcf is missing.')
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
        for column, index in self.column_to_index.items():
            assert (self.columns[index] == column)
        self.__contig_id_list = [contig['ID'] for contig in self.contig_list]
        self.__data_to_write = dict([(contig_id, {}) for contig_id in self.__contig_id_list])
        return

    @write_mode_only
    def set_samples(self, sample_list):
        """
        set_samples(sample_list) -> set the sample information from sample_list.
        """
        if self.__data_to_write != {}:
            print('WARNING: set_samples: All the previously written vcf records will be lost.')
            self.__data_to_write = dict([(contig_id, {}) for contig_id in self.__contig_id_list])
        if self.__previous is not None:
            raise ValueError('Changing sample after data flushing is not allowed.')
            return

        self.samples = copy.deepcopy(sample_list)
        self.sample_to_index = dict([(key, index) for index, key in enumerate(self.samples)])
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
            if my_dict['Number'] not in ['A', 'R', 'G', '.']:
                raise ValueError('Unsupported Number=%s in %s'%(my_dict['Number'], line))
        return my_dict

    def __read_contig_from_header(self, line):
        find_text_list = ['ID=', ',length=', ',assembly=']
        find_indices = [line.index(find_text) for find_text in find_text_list]
        my_dict = {'ID': line[find_indices[0] + len(find_text_list[0]) : find_indices[1]],
                   'length': int(line[find_indices[1] + len(find_text_list[1]) : find_indices[2]]),
                   'assembly': line[find_indices[2] + len(find_text_list[2]) : -1]}
        self.contig_list.append(my_dict)
        if my_dict['ID'] in self.__contig_id_list:
            raise ValueError('The header has duplicated contig id: %s' %my_dict['ID'])
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
                raise ValueError('A \"Flag\" can not have a value.')
            else:
                raise ValueError('Unsupported type for data conversion: %s' %my_dict['Type'])
        if my_dict['Type'] == 'String' and my_dict['Number'] == 1:
            if ',' in value:
                raise ValueError('A String of Number=1 shall not contain a comma.')
        return value

    def set_bypass_size_check_tags(self, tags):
        """
        set_bypass_size_check_tags(tags) -> set the tags in INFO FILED or FORMAT that will bypass the size check.
        tags can be a string or list of string.
        """
        if type(tags) is str:
            self.__bypass_size_check_tags = tuple([tags, ])
        else:
            self.__bypass_size_check_tags  = tuple(tags)

    def __decode_value(self, my_value, my_dict, num_alt):
        if my_dict['IsMultipleEntries']:
            splitted_value = my_value.split(',')
            self.__check_length(splitted_value, my_dict, num_alt)
            return [self.__convert_to_type(value, my_dict) for value in splitted_value]
        else:
            return self.__convert_to_type(my_value, my_dict)

    def __encode_one_value(self, value):
        # The convention is to set 0.0 and 1.0 to be '0' and '1' in the vcf record
        return str(int(value)) if value in [0.0, 1.0] else str(value)

    def __encode_values(self, my_value, my_dict, num_alt):
        self.__check_length(my_value, my_dict, num_alt)
        if my_dict['IsMultipleEntries']:
            return ','.join([self.__encode_one_value(self.__convert_to_type(v, my_dict)) for v in my_value])
        else:
            return self.__encode_one_value(self.__convert_to_type(my_value, my_dict))

    def __get_key_dict(self, key, is_info_field):
        """
        is_info_field = True if it is info field, False if it is format tag
        """
        try:
            return self.info_field_dict[key] if is_info_field else self.format_tag_dict[key]
        except KeyError:
            raise KeyError('Unknow key \"%s\" in %s: not specified in the header.' %(key, 'INFO' if is_info_field else 'FORMAT'))
            return None

    def __is_multiple_entries(self, format_dict):
        if format_dict['Number'] in [0, 1]:
            return False
        if format_dict['Number'] in ['A', 'R', 'G', '.', ]:
            return True
        if type(format_dict['Number']) is int and format_dict['Number'] > 1:
            return True
        return False

    def __check_length(self, value, format_dict, num_alt):
        if value in ['.', ['.',]] or format_dict['Number'] in ['.', 'G'] or format_dict['ID'] in self.__bypass_size_check_tags:
            # TS-14886, IR-29193
            # Also, I currently don't support the size check for 'G'
            return True

        my_len = len(value) if type(value) in [list, tuple] else 1
        expected_len = format_dict['Number']
        if format_dict['Number'] == 'A':
            expected_len = num_alt
        elif format_dict['Number'] == 'R':
            expected_len = num_alt + 1
        # Again, I currently don't support Number=G.

        if expected_len != my_len:
            if format_dict['Number'] in ['A', 'R']:
                raise IndexError('The length of %s conflicts len(ALT) = %d' %(str({format_dict['ID']: value}), num_alt))
            else:
                raise TypeError('%s conflicts the \"Number\" specified in the header.'%str({format_dict['ID']: value}))
            return False
        return True


    def __decode_format_tag_one_entry(self, tag_key, tag_value, num_alt):
        my_tag_dict = self.__get_key_dict(tag_key, False)
        return tag_key, self.__decode_value(tag_value, my_tag_dict, num_alt)

    def __decode_format_tag(self, format_keys, samples_tags, num_alt):
        splitted_format_tag_key = format_keys.split(':')
        format_tag_list = []
        for sample_idx in range(len(self.samples)):
            splitted_format_tag_value = samples_tags[sample_idx].split(':')
            if len(splitted_format_tag_key) != len(splitted_format_tag_value):
                raise ValueError('FORMAT \"%s\"and TAGS \"%s\"mismatches:'%(format_keys, samples_tags[sample_idx]))
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
                raise ValueError('The INFO FIELD \"%s\" contains more that one \"=\"' %info_text_one_entry)
                return None, None
            info_key = splitted_info_text[0]
            info_value = splitted_info_text[1]

        my_info_dict = self.__get_key_dict(info_key, True)

        my_type = my_info_dict['Type']
        if info_value is None:
            if my_type != 'Flag':
                raise TypeError('The INFO FIELD \"%s\" conflicts %s' %(info_text_one_entry, str(my_info_dict)))
            return info_key, info_value

        return info_key, self.__decode_value(info_value, my_info_dict, num_alt)


    def read_vcf_record(self, line):
        """
        read_vcf_record(line) -> dict
        Decode a text of vcf record to a dict
        """
        if ' ' in line:
            raise ValueError('A valid vcf record can not have a space: \"%s\"'%line)
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
                        'RAWLINE': line,
                        }
        except Exception as e:
            print('Error in vcf line: ' + line)
            raise type(e)(str(e) + ' happens at %s' % line)

        return vcf_dict

    def __encode_info_field(self, info_dict, num_alt):
        info_list = []
        info_keys = list(info_dict.keys())
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
            format_order = list(format_tag_dict.keys())
            format_order.sort()
        else:
            if len(format_order) != len(list(format_tag_dict.keys())):
                raise IndexError('Order of the format tags does not match the tags.')

        format_tag_list = []
        for key in format_order:
            my_format_dict = self.__get_key_dict(key, False)
            format_tag_list.append(self.__encode_values(format_tag_dict[key], my_format_dict, num_alt))

        return ':'.join(format_order), format_order,':'.join(format_tag_list)

    @write_mode_only
    def write(self, vcf_dict, is_write_raw_line = False):
        """
        write(vcf_dict, is_write_raw_line = False) -> write the vcf record to the buffer (mode = 'w' only)
        Encode vcf_dict to the text format and save it to the buffer.
        If is_write_raw_line, then write vcf_dict['RAWLINE']
        The vcf record will not be written to the file until self.flush()
        """
        my_chrom = vcf_dict['CHROM']
        my_pos = int(vcf_dict['POS'])
        if is_write_raw_line:
            vcf_text = vcf_dict['RAWLINE']
            # In case someone edited vcf_dict['CHROM'] and/or vcf_dict['POS'].
            if not vcf_text.startswith('%s\t%d\t' %(my_chrom, my_pos)):
                splitted_line = vcf_text.split('\n')
                my_chrom = splitted_line[self.column_to_index['CHROM']]
                my_pos = int(splitted_line[self.column_to_index['POS']])
        else:
            vcf_text = self.vcf_dict_to_text(vcf_dict)

        if self.__previous is not None:
            my_comp = self.compare_positions(self.__previous, {'contig': my_chrom, 'pos': my_pos})
            if my_comp > 0:
                pass
            elif my_comp == 0 and (not self.__allow_same_position):
                raise ValueError('Writing multiple vcf lines at the same position %s:%d is not allowed. Use self.allow_same_position(True) to allow.' %(contig_id, pos_key))
                return
            elif my_comp < 0:
                raise ValueError('Writing the position %s:%d that comes before the position %s:%d that has been flushed is not allowed.' %(my_chrom, my_pos, self.__contig_id_list[self.__previous['contig_idx']], self.__previous['pos']))
                return

        my_lines_at_pos = self.__data_to_write[my_chrom].setdefault(my_pos, [])
        my_lines_at_pos.append(vcf_text)
        self.__num_records_in_write_buffer += 1

    def vcf_dict_to_text(self, vcf_dict):
        """
        vcf_dict_to_text(vcf_dict) -> str
        Encode vcf_dict to the text format.
        """
        alt = ','.join(vcf_dict['ALT'])
        num_alt = len(vcf_dict['ALT'])
        info_field_text = self.__encode_info_field(vcf_dict['INFO'], num_alt)
        if len(vcf_dict['FORMAT']) != len(self.samples):
            raise IndexError('FORMAT TAG does not match the number of smaples')
            return None
        format_order = vcf_dict.get('FORMAT_ORDER', None)
        format_tag_list = []
        for format_tag_dict in vcf_dict['FORMAT']:
            my_format_order_text, my_format_order, format_tag = self.__encode_format_tag(format_tag_dict, num_alt, format_order)
            if format_order is None:
                format_order = my_format_order
            elif format_order != my_format_order:
                raise ValueError('The samples have different FORMAT.')
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
        for key, index in self.column_to_index.items():
            splitted_vcf_text[index] = my_vcf_text_dict[key]
        splitted_vcf_text += format_tag_list

        vcf_line = '\t'.join(splitted_vcf_text)
        return vcf_line


    def add_to_header(self, new_header_text):
        """
        add_to_header(new_header_txt)
        Add a new header entry to the vcf
        """
        self.__read_header_line(new_header_text)
        self.vcf_double_hash_header.append(new_header_text)


class TestTvcVcfFileReadMode:
    @staticmethod
    def write_vcf_headers(f):
        f.write(b'##contig=<ID=chr1,length=249250621,assembly=hg19>\n')
        f.write(b'##INFO=<ID=NS,Number=1,Type=Integer,Description="Number of samples with data">\n')
        f.write(b'##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Read Depth">\n')
        f.write(b'#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	5pctMM-rep1\n')
        f.seek(0)

    @classmethod
    def runTest(cls):
        t = cls()
        test_prefix = 'test_'
        test_methods = [m for m in dir(t) if callable(getattr(t, m)) and m.startswith(test_prefix)]

        for method in test_methods:
            getattr(t, method)()

    def __init__(self):
        from tempfile import NamedTemporaryFile
        self._tempfile = NamedTemporaryFile

    @property
    def tempfile(self):
        return self._tempfile

    def test_iter(self):
        # _iter_()
        with self.tempfile() as f:
            TestTvcVcfFileReadMode.write_vcf_headers(f)
            vcf = TvcVcfFile(f.name, 'w')
            try:
                for line in vcf:
                    print(line)
            except IOError as err:
                msg = str(err)
                assert 'File not open in read (r) mode for __iter__' in msg

    def test_next(self):
        # python2: next() python3: __next__()
        with self.tempfile() as f:
            TestTvcVcfFileReadMode.write_vcf_headers(f)
            vcf = TvcVcfFile(f.name, 'w')
            try:
                next(vcf)
            except IOError as err:
                msg = str(err)
                assert 'File not open in read (r) mode for __next__' in msg

    def test_reset(self):
        # reset()
        with self.tempfile() as f:
            TestTvcVcfFileReadMode.write_vcf_headers(f)
            vcf = TvcVcfFile(f.name, 'w')
            try:
                vcf.reset()
            except IOError as err:
                msg = str(err)
                assert 'File not open in read (r) mode for reset' in msg

    def test_seek(self):
        # seek()
        with self.tempfile() as f:
            TestTvcVcfFileReadMode.write_vcf_headers(f)
            vcf = TvcVcfFile(f.name, 'w')
            try:
                vcf.seek(0)
            except IOError as err:
                msg = str(err)
                assert 'File not open in read (r) mode for seek' in msg

    def test_tell(self):
        # tell()
        with self.tempfile() as f:
            TestTvcVcfFileReadMode.write_vcf_headers(f)
            vcf = TvcVcfFile(f.name, 'w')
            try:
                vcf.tell()
            except IOError as err:
                msg = str(err)
                assert 'File not open in read (r) mode for tell' in msg

    def test_set_header(self):
        # set_header
        with self.tempfile() as f:
            TestTvcVcfFileReadMode.write_vcf_headers(f)
            vcf = TvcVcfFile(f.name, 'r')
            try:
                vcf.set_header('something')
            except IOError as err:
                msg = str(err)
                assert 'File not open in write (w) mode for set_header' in msg

    def test_set_samples(self):
        # set_samples
        with self.tempfile() as f:
            TestTvcVcfFileReadMode.write_vcf_headers(f)
            vcf = TvcVcfFile(f.name, 'r')
            try:
                vcf.set_samples('something')
            except IOError as err:
                msg = str(err)
                assert 'File not open in write (w) mode for set_samples' in msg

    def test_write(self):
        # write
        with self.tempfile() as f:
            TestTvcVcfFileReadMode.write_vcf_headers(f)
            vcf = TvcVcfFile(f.name, 'r')
            try:
                vcf.write('something')
            except IOError as err:
                msg = str(err)
                assert 'File not open in write (w) mode for write' in msg

    def test_flush(self):
        # flush
        with self.tempfile() as f:
            TestTvcVcfFileReadMode.write_vcf_headers(f)
            vcf = TvcVcfFile(f.name, 'r')
            try:
                vcf.flush()
            except IOError as err:
                msg = str(err)
                assert 'File not open in write (w) mode for flush' in msg


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

    # run test with read mode
    TestTvcVcfFileReadMode.runTest()

