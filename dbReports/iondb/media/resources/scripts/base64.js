function encodeBase64(str){
	var chr1, chr2, chr3, rez = '', arr = [], i = 0, j = 0, code = 0;
	var chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/='.split('');

	while(code = str.charCodeAt(j++)){
		if(code < 128){
			arr[arr.length] = code;
		}
		else if(code < 2048){
			arr[arr.length] = 192 | (code >> 6);
			arr[arr.length] = 128 | (code & 63);
		}
		else if(code < 65536){
			arr[arr.length] = 224 | (code >> 12);
			arr[arr.length] = 128 | ((code >> 6) & 63);
			arr[arr.length] = 128 | (code & 63);
		}
		else{
			arr[arr.length] = 240 | (code >> 18);
			arr[arr.length] = 128 | ((code >> 12) & 63);
			arr[arr.length] = 128 | ((code >> 6) & 63);
			arr[arr.length] = 128 | (code & 63);
		}
	};

	while(i < arr.length){
		chr1 = arr[i++];
		chr2 = arr[i++];
		chr3 = arr[i++];

		rez += chars[chr1 >> 2];
		rez += chars[((chr1 & 3) << 4) | (chr2 >> 4)];
		rez += chars[chr2 === undefined ? 64 : ((chr2 & 15) << 2) | (chr3 >> 6)];
		rez += chars[chr3 === undefined ? 64 : chr3 & 63];
	};
	return rez;
};