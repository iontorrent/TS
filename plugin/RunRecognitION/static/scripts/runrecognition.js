function getGuruBaseUrl() {
	return 'http://torrentguru.iontorrent.com'
}

function addCommas(nStr) {
	nStr += '';
	x = nStr.split('.');
	x1 = x[0];
	x2 = x.length > 1 ? '.' + x[1] : '';
	var rgx = /(\d+)(\d{3})/;
	while (rgx.test(x1)) {
		x1 = x1.replace(rgx, '$1' + ',' + '$2');
	}
	return x1 + x2;
}

function initializeLeaderboard(leagueId, divId, rundata, lbPositionDivId) {
	$.ajax({
        url:  getGuruBaseUrl() + '/runrecognition/api/v1/leaderboard/' + leagueId + '/',
        dataType : "jsonp",
        contentType : "application/json; charset=utf-8",
        type : "GET",
        timeout: 5000,
        beforeSend: function(jqXHR, settings) {
        	$(divId).html('<div style="text-align: center; background: #ffffff;"><img src="/site_media/jquery/colorbox/images/loading.gif" alt="Loading" style="float:center"></img><p>Loading...</p></div>');
        }, 
        error: function(XHR, textStatus, errorThrown){
        	if (textStatus != 'success') {
        		$(divId).html("<div style='background: #ffffff;'>We are unable to connect to the leaderboard at this time.</div>");
        	}
        },
        success : function(data) {
            var htmlstringArray = new Array();
            htmlstringArray.push('<table class="spanner"><tr><th class="totalbar">Total Reported ');
        	htmlstringArray.push(data.cumulative_field_name);
        	htmlstringArray.push(': ');
        	htmlstringArray.push(addCommas(data.cumulative_field_value) + ' ' + data.cumulative_field_label);
        	htmlstringArray.push('</th></tr></table>');
            htmlstringArray.push('<table class="leaderboard"><tr><th scope="colgroup" class="league" colspan="4">Ion ');
            htmlstringArray.push(data.chip_type);
            htmlstringArray.push('&trade; League</th></tr><tr><th scope="col">Rank/User</th><th scope="col" class="alignright" nowrap>');
            htmlstringArray.push(data.metric_label);
            htmlstringArray.push('</th><th scope="col" class="alignright" nowrap>Run Date</th>');
            htmlstringArray.push('<th scope="col" class="alignright" nowrap>Date Submitted</th></tr>');
            if (data.entries.length == 0) {
            	htmlstringArray.push('<tr class="odd"><td colspan="4" class="noentries">There are no entries in this league yet.</td></tr>');
            } else {
	            $.each(data.entries, function(index, value) { 
	                if (index % 2 == 0) {
	                    htmlstringArray.push('<tr class="odd">');
	                } else {
	                	htmlstringArray.push('<tr>');
	                }
	                htmlstringArray.push('<td class="nopad"><table class="rankuser"><tr><td class="rank">');
	                htmlstringArray.push(value.rank);
	                htmlstringArray.push('</td><td class="avatar"><img src="/site_media/RunRecognitION/images/user.png" class="avatar16x16"/></td><td class="username">');
	                htmlstringArray.push(value.username);
	                htmlstringArray.push('</td></tr></table></td><td class="alignright">');
	                htmlstringArray.push(addCommas(value.metric_value));
	                htmlstringArray.push('</td><td class="alignright normal" nowrap>');
	                htmlstringArray.push(value.date_of_run);
	                htmlstringArray.push('</td><td class="alignright normal" nowrap>');
	                htmlstringArray.push(value.created_date);
	                htmlstringArray.push('</td></tr>');
	            });
            }
            htmlstringArray.push('<tr><th colspan="4" class="footer">&nbsp;</th></tr>');

            htmlstringArray.push('</table><div class="clear"></div>');
            var writestring = htmlstringArray.join('');
            $(divId).html(writestring);
            
            if (rundata != null && lbPositionDivId != null) {
            	if (data.entries.length == 0) {
            		$(lbPositionDivId).html('<p>Your run would place you at #<strong>1</strong> on the Leaderboard.</p>');
            	} else {
            		$.ajax({
	                    url:  getGuruBaseUrl() + '/runrecognition/api/v1/experimentrunfielddefinition/',
	                    dataType : "jsonp",
	                    contentType : "application/json; charset=utf-8",
	                    type : "GET",
	                    async: false,
	                    data: {name: data.metric_label},
	                    success : function(field_def_data) {
	                    	 user_metric_value = rundata[field_def_data.objects[0].ts_object][field_def_data.objects[0].ts_field]
	                    	 rank_found = false;
	                    	 $.each(data.entries, function(index, value) { 
	                    		 if (rank_found == false && user_metric_value >= value.metric_value) {
	                    			 $(lbPositionDivId).html('<p>Your run would place you at #<strong>' + value.rank + '</strong> on the Leaderboard.</p>');
	                    			 rank_found = true;
	                    		 }
	                    	 });
	                    	 
	                    	 if (rank_found == false && data.entries.length < data.max_size) {
	                    		 $(lbPositionDivId).html('<p>Your run would place you at #<strong>' + (data.entries.length + 1) + '</strong> on the Leaderboard.</p>');
	                    	 }
	                    }
	            	});
            	}
            }
        }
    });
}

function getRunData(experimentId) {
	runData = {};
	$.ajax({
		url : "/rundb/api/v1/results/" + experimentId + "/?format=json",
		dataType : "json",
		type : "GET",
		async : false,
		success : function(data) {
			runData['results'] = data;
		}
	});
	
	$.ajax({
        url : runData['results'].experiment,
        dataType : "json",
        type : "GET",
        async: false,
        success : function(data) {
        	runData['experiment'] = data
        }
    });
	
	$.ajax({
        url : runData['results'].qualitymetrics,
        dataType : "json",
        type : "GET",
        async: false,
        success : function(data) {
        	runData['qualitymetrics'] = data
        }
    });
	
	$.ajax({
        url : runData['results'].libmetrics,
        dataType : "json",
        type : "GET",
        async: false,
        success : function(data) {
        	runData['libmetrics'] = data
        }
    });
	return runData;
}
