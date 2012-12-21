//this file has the JavaScript for the default report

function htmlEscape(str) {
    return String(str).replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/'/g, '&#39;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/\n/g, '<br\>');
}

//string endswith
String.prototype.endsWith = function (str) {
    return (this.match(str + "$") == str)
}

function resizeiFrames(){
    //Resize the iframe blocks
    
    $(".pluginBlock:visible").each(function(){
        var height = $(this).contents().height(),
            width = $(".pluginGroupList").css("width");
        if($(this).height() != height) $(this).height(height);
        if($(this).width() != width) $(this).width(width);
    });

    $(".pluginMajorBlock:visible").each(function(){
        var height = $(this).contents().height(),
            width = parseInt($(".section").css("width"),10) - 20;
        height = height > 800 ? 800 : height;
    	if ($(this).height() != height)  $(this).height(height);
        if ($(this).width() != width)  $(this).width(width);
    });

     if ( $.browser.msie ) {
     	$('.pluginBlock, .pluginMajorBlock').css("height",400);
     }
}

//get the status of the plugins from the API
function pluginStatusLoad() {
    //init the spinner -- this is inside of the refresh button
    $('#pluginRefresh').activity({
        segments: 10,
        width: 3,
        space: 2,
        length: 3,
        color: '#252525',
        speed: 1.5,
        padding: '3',
        align: 'left'
    });
    $("#pluginStatusTable").fadeOut();
    $("#pluginStatusTable").html("");
    $.ajax({
        type: 'GET',
        url: djangoURL,
        contentType: "application/json; charset=utf-8",
        dataType: "json",
        async: false,
        success: function (data) {
        	if ($.browser.msie){
    			iframescroll = "yes";
    		}else{
    			iframescroll = "no";
    		}
            for (var i = 0; i < data.length; i++) {
                var row = "";
                //TODO: use Hogan to do this
                //Write plugin name
                row += "<div class='pluginGroup'><div class='pluginGroupHead'><div class='pluginGroupHeadInner'>";
                row += '<button class="btn plugin-collapse">-</button> ';
                row += data[i].Name;
                row += " &#8212; v";
                row += data[i].Version;
                //Write the job status
                row += '<a class="pluginLog ui-icon ui-icon-script" data-title="Log for ' + data[i].Name + '" title="Log File for ' + data[i].Name + '" href="' + webLink + '/plugin_out/' + data[i].Name + '_out/drmaa_stdout.txt">log</a>';
                row += "<strong class='pluginState'> " + data[i].State + "</strong>";
                row += "</div></div>";
                //for each file make a new row under the plugin name
                //if it ends with _block.html or _block.php then render that to a div inline
                row += '<div class="pluginGroupList" id="' + data[i].Name + '">';
                if (data[i].Files) {
                    for (var j = 0; j < data[i].Files.length; j++) {
                        if ((data[i].Files[j].endsWith("_block.html") || data[i].Files[j].endsWith("_block.php") ) && !(data[i].Major) ) {
                            row += '<div class="pluginBlock"><iframe id="' + data[i].Name + '" class="pluginBlock" src="' + webLink + '/plugin_out/' + data[i].Name + '_out/' + data[i].Files[j] + '" width="95%" frameborder="0" height="0px" scolling="'+ iframescroll + '"></iframe></div>';
                        }
                        else {
                            row += '<div class="pluginFile">&bull;<a href="' + webLink + '/plugin_out/' + data[i].Name + '_out/' + data[i].Files[j] + '"> ';
                            row += data[i].Files[j];
                            row += "</a></div>";
                        }
                    }
                }
                row += '</div>';
                $("#pluginStatusTable").append(row).find(".plugin-collapse:last").click(function(){
                    $(this).text($(this).text() == '-' ? '+' : '-');
                    $(this).closest(".pluginGroup").find(".pluginGroupList").slideToggle(250);
                    return false;
                });
            }
        },
        error: function (msg) {
            $("#pluginStatusTable").text("Failed to get Plugin Status: " + msg);
        }
    }); //for ajax call
    $("#pluginStatusTable").fadeIn();
    $('#pluginRefresh').activity(false);

}

function progress_load(){
    $.ajax({
	    type: 'GET',
	    url: reportAPI,
	    contentType: "application/json; charset=utf-8",
	    dataType: "json",
	    async: false,
	    success: function (data) {
	    	$("#progress_message").html(data.status);
	    	
			if (data.status.indexOf("Completed") == 0){
				clearInterval($(document).data('report_progress'));
				if (data.status === "Completed") {
					$("#progress_message").html("The report has been completed. The page will reload in 5 seconds");
					setTimeout(function(){location.reload()}, 5000);
				}
			}
	    },
		error: function (msg) {
	    	$("#progress_message").html("Error checking status");
		}
	});

}

$(document).ready(function(){

	//get the report PK so we can do stuff!	
	djangoPK = $("#report").data("pk");
	webLink = $("#report").data("web");
	reportAPI = "/rundb/api/v1/results/" + djangoPK + "/";
	djangoURL = "/rundb/api/v1/results/" + djangoPK + "/pluginresults/";

    //keep this incase we want to bind to something
	$('#barcodes tr').click(function() {
		var id =  $(this).find('td:first').text().trim(); 
    });

	$.colorbox.close = function(){
		//TODO THIS IS HUGE HACK fix it for 3.0
		//I can't close colorbox so I reload the page
		location.reload();
	};

	//remove and turn off the progres indicator 
	$("#close_progress").click(function(){
		$("#backdrop_progress, #progress").remove();
		clearInterval($(document).data('report_progress'));
	});

	//proton 
	protonData = $("#report").data("php") + " #isMap"; 
	$("#proton").load(protonData);

	//load the tabs
    var q30_quality = $("#q30_quality").data("percent");
    $("#q30_quality").strength(q30_quality, 'undefined', q30_quality, 
        'Sequence >= Q30');

	//init modal
	$('#plugin-modal').modal({
	    backdrop: true,
	    show: false
	});

	//report list
	$("#resultList").change(function(){
		window.location = "/report/" + $(this).val();
	});

	//plan popup modal
	$('#review_plan').click(function(e){
		e.preventDefault();
		pk = $(this).data("pk");
		url = "/plan/reviewplan/" + pk + "/";
		
		$('body #modal_review_plan').remove();
		$.get(url, function(data) {
		  	$('body').append(data);
		    $( "#modal_review_plan" ).modal("show");
		    return false;
		}).done(function(data) { 
	    	// $(that).trigger('remove_from_project_done', {values: e.values});
		})
	    .fail(function(data) {
	    	$('#error-messages').empty();
	    	$('#error-messages').append('<p class="error">ERROR: ' + data.responseText + '</p>');     	 
	    })
	    .always(function(data) { /*console.log("complete:", data);*/ });		
	});   

	//do the initial plugin status load
	pluginStatusLoad();

    $.ajax({
    type: 'GET',
    url: djangoURL + "?major=true",
    contentType: "application/json; charset=utf-8",
    dataType: "json",
    async: false,
    success: function (data) {
    		if ($.browser.msie){
    			iframescroll = "yes";
    		}else{
    			iframescroll = "no";
    		}
    		
	        for (var i = 0; i < data.length; i++) {
	        	//if there is a block, show it
	        	if (data[i].Files.length > 0){
	        		$("#major_blocks").append('<h2>' +  data[i].Name + ' (' + data[i].Version + ') </h2> <div class="section"><iframe scrolling="'+ iframescroll +'" id="' + data[i].Name + '" class="pluginMajorBlock" src="' + webLink + '/plugin_out/' + data[i].Name + '_out/' + data[i].Files[0] + '" frameborder="0" height="0px" ></iframe></div>');
	        	}
	        }
    	}
	});

	//TODO : Rewrite with bootstrap
	//provide the SGE log for plugins
	$('.pluginLog').live("click", function (e) {
		e.preventDefault();
	    var url = $(this).attr("href");
	    var title = $($(this)).data("title");
	    // load remote content
	    var logParent = $($(this).parent());
	    logParent.activity({
	        segments: 10,
	        width: 3,
	        space: 2,
	        length: 2.5,
	        color: '#252525',
	        speed: 1.5,
	        padding: '3'
	    });
	    $.get(url, function (responseText) {
	        logParent.activity(false);
	        var $modal_plugin_log = $('#modal_plugin_log'); 
	        $modal_plugin_log.find('.modal-body').html('<pre class="log">' + htmlEscape(responseText) + '</pre>');
	        $modal_plugin_log.find('.modal-header h3').text(title);
	        $modal_plugin_log.modal('show');
	    });
	    //prevent the browser to follow the link
	    return false;
	});
	$("#pluginRefresh").click(function () {
	    pluginStatusLoad()
	});

	//the plugin launcher
	$("#pluginDialogButton").click(function () {
	    //open the dialog

    	$('#plugin-modal').modal('show');

	    $("#modal-body").html("<div id='pluginLoad'></div><div id='pluginList'></div>");
	    $("#pluginLoad").html("<span>Loading Plugin List <img src='/site_media/jquery/colorbox/images/loading.gif'></img></span>");

	    //get the list of plugins from the API
	    $.ajax({
	        url: '/rundb/api/v1/plugin/?selected=true&limit=0&format=json&order_by=name',
	        dataType: 'json',
	        type: 'GET',
	        async: false,
	        success: function (data) {
	            var items = [];
	            if (data.objects.length == 0) {
	                $("#pluginLoad").html("");
	                $("#pluginList").html("<p> There are no plugins what are enabled </p>");
	                return false;
	            }
	            $("#pluginList").html('<ul id="pluginUL" class="expandable"></ul>');
	            plugins = data.objects.sort(function (a, b) {
	                return a.name.toLowerCase() > b.name.toLowerCase() ? 1 : -1;
	            });
	            //build the query string in a way that works with IE7 and IE8
	            plugin_ids = "";
	            $.each(plugins, function (count, value) {
	                plugin_ids += value.id;
	                if (count + 1 != plugins.length) {
	                    plugin_ids += ";";
	                }
	            });
	            //get plugin metadata
	            $.ajax({
	                url: "/rundb/api/v1/plugin/set/" + plugin_ids + "/type/?format=json",
	                success: function (plugin_types) {
	                    for (var i = 0; i < plugins.length; i++) {
	                        val = plugins[i];
	                        data = plugin_types[val.id];
	                        if (data.input != undefined) {
	                            $("#pluginUL").append('<li data-id="' + val.id + '" class="plugin_input_class" id="' + val.name + '_plugin"><a href="' + data.input + '?report=' + djangoPK + '" class="plugin_link colorinput">' + val.name + '</a>' + '<span>' + " &#8212; v" + val.version + '</span></li>');
	                        }
	                        else {
	                            $("#pluginUL").append('<li data-id="' + val.id + '" class="plugin_class" id="' + val.name + '_plugin"><a href="#pluginDialog">' + val.name + '</a>' + '<span>' + " &#8212; v" + val.version + '</span></li>');
	                        }
	                    }
	                },
	                async: false,
	                type: 'GET',
	                dataType: 'json'
	            });

	            $("#pluginLoad").html("");
	            $("#modal-header").html('Select a plugin');
	            $("#pluginList").show();
	        }
	    });

		$(".plugin_link").colorbox({
            width: "90%",
            height: "90%",
            iframe: true
        });

	    //now the the for each is done, show the list
	    $(".plugin_input_class").die("click");
	    $(".plugin_input_class").live("click", function () {
	        $('#plugin-modal').modal('hide')
	    });
	    $(".plugin_class").die("click");
	    $(".plugin_class").live('click', function () {
	        //get the plugin id
	        var id = $(this).data('id');
	        var pluginName = $(this).attr("id");
	        //get the plugin name
	        pluginName = pluginName.substring(0, pluginName.length - 7)
	        //build the JSON to post
	        pluginAPIJSON = {
	            "plugin": [pluginName]
	        };
	        pluginAPIJSON = JSON.stringify(pluginAPIJSON);
	        $.ajax({
	            type: 'POST',
	            url: djangoURL,
	            async: true,
	            contentType: "application/json; charset=utf-8",
	            data: pluginAPIJSON,
	            dataType: "json",
	            beforeSend: function () {
	                $("#pluginList").html("");
	                $("#pluginLoad").html("<span>Launching Plugin " + pluginName + " <img src='/site_media/jquery/colorbox/images/loading.gif'></img></span>");
	            },
	            success: function () {
	                $('#plugin-modal').modal('hide')
	                pluginStatusLoad();
	            }
	        });
	    });
	});

	//if this is a print view, unroll the tabs, and reformat the page as needed
	if ($.QueryString["no_header"]){
		$(".tab-pane").each(function(){
			$(this).removeClass("tab-pane");
			$(this).addClass("tabBox");
			$(this).prepend("<h2>" + $(this).data("title") + "</h2><hr/>");
		});
		$(".nav,.sub-nav,.page-nav,.btn, #resultSet,.footer,#resultSet,#OutputFiles").hide();
		$(".main-push").remove();
		$('.main').css({background: '-webkit-gradient(linear, 0% 0%, 100% 100%, from(#ffffff), to(#ffffff))'});
		$(".header").hide();
		$("#nameRow").removeClass("span6").addClass("span12");
		$("#beadfind, #basecaller, #readlength").removeClass("span3").addClass("span4");
		$("#alignMap, #rawAligned").removeClass("span3").addClass("span6");

		$(".tabBox").css({
		  "border": "3px black solid",
		  "margin": "20px",
		});
		$(".pluginGroup").css({
		  "border": "1px black solid",
		  "margin": "10px"
		});

		$(".unaligned .well").css("min-height", "390px");
		$(".aligned .well").css({"min-height" : "370px" , "height" : "370px"});

		//remove the shadows around the content
		$(".content").css({"box-shadow" : "0px 0px",
						   "border-radius": "0px",
						   "border" : "0px"
						});

	}else{

		//Only do these things for the non print view

		//try to resize the plugin block all the time
		//Once a second is Plenty for this.

		resizeIntervalTimer = setInterval(function(){ resizeiFrames() }, 1000); 

		//init the Kendo Grids
		$("#barcodes").kendoGrid({
	        dataSource: {
	        	pageSize: 10
	        },
	        height: 'auto',
	        groupable: false,
	        scrollable: false,
	        selectable: false,
	        sortable: false,
	        pageable: true
	    });	

		$("#file_table").kendoGrid({
	        dataSource: {
	                pageSize: 10
	        },
	        height: 'auto',
	        groupable: false,
	        scrollable: false,
	        selectable: false,
	        sortable: false,
	        pageable: false
	    });
	    
	    $("#test_fragments").kendoGrid({
	        dataSource: {
	                pageSize: 10
	        },
	        height: 'auto',
	        groupable: false,
	        scrollable: false,
	        selectable: false,
	        sortable: false,
	        pageable: true
	    });

	}


});
