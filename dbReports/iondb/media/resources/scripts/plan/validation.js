/**
	Regex test function for Template Name, and Plan Names
*/
function is_valid_chars(_string) {
	if (_string.length > 0) {
		return new RegExp(/^[a-zA-Z0-9-_\.\s]+$/).test(_string);
	}
	return true;
}

/**
	Check max length
*/
function is_valid_length(_string, max_length) {
	return _string.length <= max_length;
}

/**
	Make sure leading characters are not a dot or a dash or an underscor
*/
function is_valid_leading_chars(_string) {
	if (_string.length > 0) {
		return _string.search(/[\.\_\-]/) != 0;
	}
	return true;
}

/**
	Check percentage value
*/
function is_valid_percentage(_string) {
	var intValue = parseInt(_string);

	return ((intValue >= 0) && (intValue <= 100));
}
