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
        url:  getGuruBaseUrl() + '/runrecognition/api/v1/leaderboardleague/',
        data: {'chip_type': leagueId},
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
        success : function(leagues) {
        		var league = leagues.objects[0];
        		getAddLeaderboard($(divId), league, rundata, lbPositionDivId);
        }
    });
}

function addChipTypeToId(objJqRef, league, other){
	objJqRef.attr("id", objJqRef.attr("id") + league.chip_type);
	if (other != null) {
		objJqRef.attr("id", objJqRef.attr("id") + other);
	}
}

function getAddLeaderboard(parentJqRef, league, rundata, lbPositionDivId){
	$.ajax({
        url:  getGuruBaseUrl() + "/runrecognition/api/v1/leaderboard/" + league.chip_type,
        dataType : "jsonp",
        contentType : "application/json; charset=utf-8",
        type : "GET",
        timeout: 5000,
        success : function(leaderboard) {
	        	$.get("/pluginMedia/RunRecognitION/html/leaderboard_template.html",
	              function(rrTemplateHtmlString) {
	        			parentJqRef.empty();
	        			parentJqRef.html(rrTemplateHtmlString);
	        			var rrTemplateJqRef = parentJqRef;
	                	var holder = rrTemplateJqRef.find("#large_templates #largeWidget").clone(true, true);
	                	addChipTypeToId(holder, league, null);
	                	addLargeLeaderboard(rrTemplateJqRef, holder, league, leaderboard);
	                	parentJqRef.append(holder);
	                	getfillInPosition(leaderboard, rundata, lbPositionDivId);
	              });
    		}
    });
}


function addLargeLeaderboard(rrTemplateJqRef, holder, league, leaderboard){
	var topPart = holder.find("#topPart");
	addChipTypeToId(topPart, league, null);
	fillLargeTop(rrTemplateJqRef, topPart, league, leaderboard);
	addPointsDisclaimer(topPart, leaderboard)
	holder.append(topPart);
	
	var middlePart = holder.find("#middlePartHolder");
	addChipTypeToId(middlePart, league, null);
	fillLargeMiddle(rrTemplateJqRef, middlePart, leaderboard);
	
	holder.append(middlePart);
	holder.append($("<div>").attr("class", "clear"));
	holder.append($("<br>"));
}

function fillLargeMiddle(rrTemplateJqRef, middleHolderJqRef, data){
	middleHolderJqRef.empty();
	middleHolderJqRef.append(rrTemplateJqRef.find("#large_templates #largeWidget #middlePart").clone(true, true));
	middleJqRef = middleHolderJqRef.find("#middlePart");
	addChipTypeToId(middleJqRef.find("#middleChipType"), data, null);
	
	middleJqRef.find("#middleMetricLabel").text(data.metric_label);
	addChipTypeToId(middleJqRef.find("#middleMetricLabel"), data, null);
	
	middleJqRef.find("#dataRow").remove();
	if (data.entries.length == 0) {
		var emptyRow = $("<tr>")
		var emptyD = $("<td>").attr("colspan", "4").attr("class", "noentries");
		emptyD.text("There are no entries in this league yet.");
		emptyRow.append(emptyD);
		middleJqRef.append(emptyRow);
	} else {
		$.each(data.entries, 
				function(index, value){
					var filledRow = rrTemplateJqRef.find("#large_templates #largeWidget #middlePart #dataRow").clone(true, true);
					fillLargeRow(filledRow, index, value, data);
					middleJqRef.append(filledRow);
					addChipTypeToId(filledRow, data, "_" + index);
				});
	}
	
	var bottomRow = $("<tr>");
	var bottomD = $("<td>").attr("colspan", "4").attr("class", "footer");
	bottomD.text(rrTemplateJqRef.find("#large_templates #nbspLarge").text());
	bottomRow.append(bottomD);
	middleJqRef.append(bottomRow);
	addChipTypeToId(middleJqRef, data, null);
}

function fillLargeRow(rowJqRef, index, rowData, leaderboard){
	rowJqRef.find("#avatarImg").attr('src', '/pluginMedia/RunRecognitION/images/user.png');
	rowJqRef.find("#usernameLink").text(rowData.username);
	rowJqRef.find("#rowMetricValue").text(addCommas(rowData.metric_value));
	rowJqRef.find("#rowDateOfRun").text(rowData.date_of_run);
	rowJqRef.find("#rowCreatedDate").text(rowData.created_date);
	rowJqRef.find("#metricRank").text(index + 1);
	
	addChipTypeToId(rowJqRef.find("#rankValue"), leaderboard, "_" + index);
	addChipTypeToId(rowJqRef.find("#avatarImg"), leaderboard, "_" + index);
	addChipTypeToId(rowJqRef.find("#usernameLink"), leaderboard, "_" + index);
	addChipTypeToId(rowJqRef.find("#rowMetricValue"), leaderboard, "_" + index);
	addChipTypeToId(rowJqRef.find("#rowDateOfRun"), leaderboard, "_" + index);
	addChipTypeToId(rowJqRef.find("#rowCreatedDate"), leaderboard, "_" + index);
	addChipTypeToId(rowJqRef.find("#metricRank"), leaderboard, "_" + index);
}


function fillLargeTop(rrTemplateJqRef, topJqRef, league, leaderboard) {
	var selectJqRef = topJqRef.find("#leaderboardSelect");
	var optionText = leaderboard.metric_label;
	var optionJqRef = $("<option>");
	optionJqRef.val("/runrecognition/api/v1/leaderboard/" + leaderboard.chip_type);
	optionJqRef.text(optionText);
	selectJqRef.append(optionJqRef);
	
	$.each(league.alt_leagues, function(index, value){
		optionText = value.metric.name;
		optionJqRef = $("<option>");
		optionJqRef.val("/runrecognition/api/v1/alternativeleaderboard/" + value.id);
		optionJqRef.text(optionText);
		selectJqRef.append(optionJqRef);
	});
	
	selectJqRef.change(function(){
		switchLargeLeaderboard(rrTemplateJqRef, league, $(this).children(":selected").val());
	});
	setLargeLeaderboardSize(topJqRef, leaderboard);
	addChipTypeToId(selectJqRef, league, null);
}

function setLargeLeaderboardSize(topJqRef, leaderboard){
	var sizeText = "";
	if (leaderboard.cumulative_field_value != null) {
		sizeText = addCommas(leaderboard.cumulative_field_value) + " " + leaderboard.cumulative_field_label + ".";
	}
	var sizeJqRef = topJqRef.find("#leaderboardSize" + leaderboard.chip_type);
	if (sizeJqRef.length == 0) {
		sizeJqRef = topJqRef.find("#leaderboardSize");
		addChipTypeToId(topJqRef.find("#leaderboardSize"), leaderboard, null);
	}
	sizeJqRef.text(sizeText);
}

function addPointsDisclaimer(topJqRef, leaderboard){
	var sizeJqRef = topJqRef.find("#leaderboardSize" + leaderboard.chip_type);
	var disclaimerDivJqRef = $("<div>");
	disclaimerDivJqRef.text("*only '" + leaderboard.metric_label + "' rankings qualify for run recognition points");
	disclaimerDivJqRef.attr("style", "float: right; font-size: .4em;");
	disclaimerDivJqRef.insertAfter(sizeJqRef);
}

function switchLargeLeaderboard(rrTemplateJqRef, parentLeague, leaderboardResourceId){
	$.ajax({
        url:  getGuruBaseUrl() + leaderboardResourceId,
        dataType : "jsonp",
        contentType : "application/json; charset=utf-8",
        type : "GET",
        async: false,
        success : function(data) {
        		var middleJqRef = $("#largeWidget" + parentLeague.chip_type + " #middlePartHolder" + parentLeague.chip_type);
        		fillLargeMiddle(rrTemplateJqRef, middleJqRef, data);
        		
        		var topJqRef = $("#largeWidget" + parentLeague.chip_type + " #topPart" + parentLeague.chip_type);
        		setLargeLeaderboardSize(topJqRef, data);
    		}
    });	
}

function getfillInPosition(data, rundata, lbPositionDivId){
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
                		fillInPosition(data, rundata, lbPositionDivId, field_def_data);
                }
        	});
    	}
    }	
}

function fillInPosition(data, rundata, lbPositionDivId, field_def_data){
	 var user_metric_value = rundata[field_def_data.objects[0].ts_object][field_def_data.objects[0].ts_field]
	 var rank_found = false;
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