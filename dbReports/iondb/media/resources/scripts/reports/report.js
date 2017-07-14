//this file has the JavaScript for the default report

api_plugin_show_url = "/rundb/api/v1/plugin/show/";

function htmlEscape(str) {
  // color-friendly: do not remove this comment -- plugins may search for it to determine the capability
  // The last two substitutions allow escaped span tags
  // in this form: {span style=*...*} ... {/span}
  return str
    .replace(/&/g, '&amp;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/\n/g, '<br/>')
    .replace(/\{span([^}]+)\}/g, function (match, attrs) {
      return '<span' + attrs.replace(/\*/g, '"') + '>';
    })
    .replace(/\{\/span\}/g, '</span>')
    .replace(/\u001b?\[30;43m([^\u001b\n]+)\u001b?\[0?m/g, '<span style="background-color: #ffce54">$1</span>');
}

//string endswith
String.prototype.endsWith = function (str) {
    return (this.match(str + "$") == str);
};

function resizeiFrames() {
    //Resize the iframe blocks

    $("iframe.pluginBlock:visible").each(function () {
        var height = $(this).contents().height(),
            width = $(this).parent().css("width");
        if ($(this).height() != height) $(this).height(height);
        if ($(this).width() != width) $(this).width(width);
    });

    $("iframe.pluginMajorBlock:visible").each(function () {
        var height = $(this).contents().find("body").height() + 20;
        //console.log($(this).attr("id") + " " + height);
        var width = parseInt($(".section").css("width"), 10) - 20;
        if ($(this).height() != height) $(this).height(height);
        if ($(this).width() != width)  $(this).width(width);
    });
}

function update_plugin_show(controls) {
    $.ajax({
        type: 'POST',
        url: api_plugin_show_url,
        data: JSON.stringify(controls),
        contentType: 'application/json',
        datType: 'json'
    });
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

    $.ajax({
        type: 'GET',
        url: pluginresultsAPI,
        contentType: "application/json; charset=utf-8",
        dataType: "json",
        async: false,
        success: function (data) {
            console.log(data);
            $('#major_blocks').html('');
            iframescroll = ($.browser.msie) ? "yes" : "no";
            for (var i = 0; i < data.length; i++) {
                if(data[i].Major) {
                    var name_row = plugin_major_template(data[i]);
                    var majorblock = $(name_row).appendTo("#major_blocks");
                    var plugin_links = majorblock.find(".plugin_links")
                    for (var o = 0; o < data[i].Files.length; o++) {
                        var f = data[i].Files[o];
                        if (f.indexOf('_block')==-1) {
                            plugin_links.append('<a href="' + data[i].URL + f + '">' + f + '</a>');
                        }
                    }
                    if (data[i].Files.length == 0) {
                        majorblock.append('<div class="section">No plugin output at this time.</div>');
                    }
                    for (var j = 0; j < data[i].Files.length; j++) {
                        var f = data[i].Files[j];
                        if (f.indexOf('_block')>-1) {
                            majorblock.append('<div class="section"><iframe scrolling="' + iframescroll + '" id="' + data[i].Name + '" class="pluginMajorBlock" src="' + data[i].URL + f + '" frameborder="0" ></iframe></div>');
                        }
                    }
                }
            }
        },
        error: function (event, request, settings) {
            $('#major_blocks').text('Failed to get Plugin Status');
            console.log("Error fetching" + settings.url + ": " + event.responseText);
        }
    });
    $("#pluginStatusTable").fadeOut();
    $("#pluginStatusTable").html("");
    $.ajax({
        type: 'GET',
        url: pluginresultsAPI,
        contentType: "application/json; charset=utf-8",
        dataType: "json",
        async: false,
        success: function (data) {
            table_plugin_names = [];
            for (var i = 0; i < data.length; i++) {
                //console.log(data[i]);
                var row = plugin_status_template(data[i]);
                var name = data[i].Name;
                table_plugin_names.push(name);
                (function (name) {
                    $("#pluginStatusTable").append(row);
                    if (row.indexOf('plugin-collapse') > -1){
                        $("#pluginStatusTable").find(".plugin-collapse:last").click(function () {
                            $(this).text($(this).text() == '-' ? '+' : '-');
                            var block = $(this).closest(".pluginGroup").find(".pluginGroupList");
                            var is_visible = block.is(":visible");
                            var control = {};
                            control[name] = !is_visible;
                            update_plugin_show(control);
                            block.slideToggle(250);
                            return false;
                        });
                    }
                })(name);
            }
        },
        error: function (msg) {
            $("#pluginStatusTable").text("Failed to get Plugin Status");
        }
    }); //for ajax call
    $("#pluginStatusTable").fadeIn();
    $('#pluginRefresh').activity(false);
}

function progress_load() {
    $.ajax({
        type: 'GET',
        url: reportAPI,
        contentType: "application/json; charset=utf-8",
        dataType: "json",
        async: false,
        success: function (data) {
            $("#progress_message").html(data.status);
            if (data.status.indexOf("Completed") === 0
                || data.status === "No Live Beads") {
                clearInterval($(document).data('report_progress'));
                var reload_interval_s = 5; // Seconds until a reload
                $("#progress_message").html(
                    "The report has been completed. "
                    + "The page will reload in "
                    + reload_interval_s + " seconds");
                setTimeout(function () {
                    location.reload();
                }, reload_interval_s * 1000);
            }
        },
        error: function (msg) {
            $("#progress_message").html("Error checking status");
        }
    });
}

function start_iru(launchObj){
    //start the iru upload based on the upload type selection
    console.log(launchObj);
    pluginAPIJSON = { "plugin": ["IonReporterUploader"], "pluginconfig": launchObj };
    pluginAPIJSON = JSON.stringify(pluginAPIJSON);

    $.blockUI();

    var iru_launch = $.ajax({
        type: 'POST',
        url: pluginURL,
        contentType: "application/json; charset=utf-8",
        data: pluginAPIJSON,
        dataType: "json"
    });

    iru_launch.done(function (data) {
        $.unblockUI();
        //TODO, also so the plugin list refresh function
        bootbox.alert("Upload to Ion Reporter has started.", function() {

        });
    });

    iru_launch.fail(function (data) {
        $.unblockUI();
        alert("Failed to start IRU upload");
    });
}


function ktmpl(selector){
    // Returns a kendo.template, even if the element is not found from the given
    // selector. If the element is not found then an empty template is returned.
    var el = $(selector);
    return (el.length) ? kendo.template(el.html()): kendo.template("");
}


$(document).ready(function () {
    plugin_status_template = ktmpl("#pluginStatusTemplate");
    plugin_major_template = ktmpl("#pluginMajorBlockTemplate");
    plugin_dropdown_template = ktmpl("#pluginDropdownTemplate");

    $(".dropdown-menu > li.disabled > a").click(function(event) {
      event.preventDefault();
    });

    //get the report PK so we can do stuff!
    djangoPK = $("#report").data("pk");
    webLink = $("#report").data("web");
    reportAPI = "/rundb/api/v1/results/" + djangoPK + "/";
    //pluginresultsAPI = "/rundb/api/v1/pluginresult/?result=" + djangoPK + "&limit=0&order_by=plugin__name";
    pluginresultsAPI = "/rundb/api/v1/results/" + djangoPK + "/pluginresults/";
    pluginURL = "/rundb/api/v1/results/" + djangoPK + "/plugin/";

    $.ajax({
        url: "/rundb/api/v1/results/" + djangoPK + "/scan_for_orphaned_plugin_results",
        type: "GET"
    })

    //If IRU is has configs then add them to a dropdown button for easy uploading
    var iru_xhr = $.ajax({
        url: "/rundb/api/v1/plugin/IonReporterUploader/extend/configs/?format=json",
        type: "GET"
    }).done(function (data) {
        var template = kendo.template($("#iru-list-tmpl").html());

        var result = template(data); //Execute the template
        $("#iru-list").html(result); //Append the result

        $('#iru-button').on("click", ".iru-account", function (event) {
            event.preventDefault();
            var id = $(this).data("id");
            var isAsPlanned = $(this).data("value");
            var launchObj;
            $.each(data, function (k, v) {
                if (v["id"] === id) {
                    launchObj = v;
                }
            });
            launchObj["iru_qc_option"] = "no_check";

            console.log("launch obj", launchObj);

            //check to see if previous instance of plugin exists and still running
            var alreadyGoing = false;
            var pluginresult='';
            $.ajax({
                type: 'GET',
                url: reportAPI,
                contentType: "application/json; charset=utf-8",
                async: false,
                dataType: "json"
            }).done(function(result){
                var data = result.pluginresults;
                pluginresult = $.map(data,function(pr) {
                    if(pr.pluginName === "IonReporterUploader"){return pr;}
                });
            });

            if(pluginresult.length>0){
                $.ajax({
                    dataType: "json",
                    contentType: "application/json",
                    url: "/rundb/api/v1/plugin/IonReporterUploader/extend/lastrun/",
                    type: "POST",
                    async: false,
                    data: JSON.stringify({'pluginresult':pluginresult[0]})
                }).done(function(data){
                    console.log('lastrun', data);
                    alreadyGoing = data.in_progress;
                });
            }

            if (alreadyGoing) {
                uploadMsg = "<div class='text-error'>WARNING Are you sure you want to upload to Ion Reporter, there is an upload already in progress?</div>";
            } else {
                uploadMsg = "Are you sure you want to upload to Ion Reporter?";
            }

            var upload = "";
            var uploadOptions = "";
            if (isAsPlanned) {
                launchObj["launchoption"] = "upload_and_launch";
    			var irActName = $(this).data("iraccountname");
                var report_uri = $(this).data("uri");

                // Generate the IR short name, pattern match as much as possbile
                irActName_short = irActName.replace(/(\(Version\: \d(.*?) \| User\: (.*?) \| Org\: (.*?)\)$)/, "");
                uploadMsg = uploadMsg.replace(/\?/,"(" + $.trim(irActName_short) + ") ? ")
                console.log("Report URI:", report_uri, "IR AccountName:", irActName);
                uploadOptions = [
                    {
                        "label": "Yes",
                        "callback": function () {
                            launchObj["upload"] = "both";
                            start_iru(launchObj);
                        }
                    },
                    {
                        "label": "No",
                        "callback": function () {
                            upload = false;
                        }
                    },
                    {
                        "label" : "Review-Plan",
                        "callback": function () {
                            upload = false;
                            $('body #modal_review_plan').remove();
                            bootbox.dialog(uploadMsg,uploadOptions);

                            $.get(report_uri,function (data) {
                                 $('body').append(data);
                                 $("#modal_review_plan").modal("show");
                                 return false;
                            }).done(function (data) {
                                 //consol.log("Review the plan");
                            }).fail(function (data) {
                                 $('#error-messages').empty().show();
                                 responseText = "Unable to open the plan for review";
                                 $('#error-messages').append('<p class="alert error">ERROR: ' + responseText + '</p>');
                            }).always(function (data) {
                                 /*console.log("complete:", data);*/
                            });
                         }
                    }

                ];
            }
            else {
               launchObj["launchoption"] = "upload_only";
               uploadOptions = [
                    {
                        "label": "Upload just BAM",
                        "class": "btn-primary",
                        "callback": function () {
                            launchObj["upload"] = "bam_only";
                            start_iru(launchObj);
                        }
                    },
                    {
                        "label": "Upload just VCF",
                        "class": "btn-primary",
                        "callback": function () {
                            launchObj["upload"] = "vcf_only";
                            start_iru(launchObj);
                        }
                    },
                    {
                        "label": "Upload BAM & VCF",
                        "class": "btn-primary",
                        "callback": function () {
                            launchObj["upload"] = "both";
                            start_iru(launchObj);
                        }
                    },
                    {
                        "label": "Cancel",
                        "callback": function () {
                            upload = false;
                        }
                    }
                ];
            }
            bootbox.dialog(uploadMsg,uploadOptions);

        });
    });

    //remove the focus from the report dropdown
    $("#resultList").blur();

    $("#resultList").chosen();

    //keep this incase we want to bind to something
    $('#barcodes tr').click(function () {
        var id = $(this).find('td:first').text().trim();
    });
    //keep this incase we want to bind to something
    $('#CA_barcodes tr').click(function () {
        var id = $(this).find('td:first').text().trim();
    });

    $.colorbox.close = function () {
        //TODO THIS IS HUGE HACK fix it for 3.0
        //Now it is an even bigger hack. Plugins have hard coded calls to this function to close their modals.
        //We are now not using colorbox, but because this call refreshed the page, plugin code still works without
        //modification.
        //I can't close colorbox so I reload the page
        location.reload();
    };

    //remove and turn off the progress indicator
    $("#close_progress").click(function () {
        $("#backdrop_progress, #progress").remove();
        clearInterval($(document).data('report_progress'));
    });

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
    $("#resultList").change(function () {
        window.location = "/report/" + $(this).val();
    });

    //plan popup modal
    $('#review_plan').click(function (e) {
        e.preventDefault();
        $('#error-messages').hide().empty();
        var url = $(this).attr("href");

        $('body #modal_review_plan').remove();
        $.get(url,function (data) {
            $('body').append(data);
            $("#modal_review_plan").modal("show");
            return false;
        }).done(function (data) {
            // $(that).trigger('remove_from_project_done', {values: e.values});
        }).fail(function (data) {
            $('#error-messages').empty().show();
            $('#error-messages').append('<p class="alert error">ERROR: ' + data.responseText + '</p>');
        }).always(function (data) {
            /*console.log("complete:", data);*/
        });
    });


    $("#copy_plan").click(function (e) {
        $('body').css("cursor", "wait");
        e.preventDefault();
        $('#error-messages').hide().empty();
        var busyDiv = '<div class="myBusyDiv"><div class="k-loading-mask" style="width:100%;height:100%"><span class="k-loading-text">Loading...</span><div class="k-loading-image"><div class="k-loading-color"></div></div></div></div>';
        $('body').prepend(busyDiv);

        pk = $(this).data("pk");
        url = "/plan/planned/" + pk + "/copy/";

        $('body #modal_plan_wizard').remove();
        $('body #modal_plan_run').remove();
        $.get(url,function (data) {
            $('body').append(data);

            setTab('#ws-1');
            $("#modal_plan_wizard").modal("show");
            return false;
        }).done(function (data) {
            console.log("success:", url);
            // $(that).trigger('remove_from_project_done', {values: e.values});
        }).fail(function (data) {
            $('body').css("cursor", "default");
            $('.myBusyDiv').empty();
            $('body').remove('.myBusyDiv');

            $('#error-messages').empty().show();
            $('#error-messages').append('<p class="error">ERROR: ' + data.responseText + '</p>');
            console.log("error:", data);

        }).always(function (data) {/*console.log("complete:", data);*/
            $('body').css("cursor", "default");
            $('.myBusyDiv').empty();
            $('body').remove('.myBusyDiv');
            delete busyDiv;
        });
    });

    // Data Management actions popup
    $('#dm_actions').click(function (e) {
        e.preventDefault();
        $('body #modal_dm_actions').remove();
        $.get($(this).attr('href'), function (data) {
            $('body').append(data);
            $("#modal_dm_actions").modal("show");
        });
        return false;
    });

    //do the initial plugin status load
    pluginStatusLoad();

    // Remove plugin result (post delete trigger will delete content)
    // confirm that it should be deleted
    $('.pluginRemove').live("click", function (e) {
        e.preventDefault();
        var url = $(this).attr("href");
        var parent = $(this).closest('div.pluginGroup');
        // TODO: Are you sure?
        $.ajax({
            type: 'DELETE',
            url: url,
            async: true,
            beforeSend: function () {
                parent.animate({'backgroundColor': '#fb6c6c'}, 300);
            }
        }).fail(function (msg) {
            parent.append('<div class=" alert alert-error" data-dismiss="alert">Failed to remove plugin result.</div>');
            parent.fadeOut(3000, function () {
                parent.remove();
            });
        }).done(function () {
            parent.append('<div class="alert alert-info" data-dismiss="alert">Successfully removed plugin result.</div>');
            parent.fadeOut(3000, function () {
                parent.remove();
            });
        });
        return false;
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
        var $modal_plugin_log = $('#modal_plugin_log');
        $modal_plugin_log.find('.modal-body').html('')
        $modal_plugin_log.find('.modal-header h3').text(title);
        $.get(url, function (responseText) {
            $modal_plugin_log.find('.modal-body').html('<pre class="log">' + htmlEscape(responseText) + '</pre>');
            var keyupFunc = function (e) {
              if (e.keyCode == 27) { // Escape key
                 $('body').unbind('keyup', keyupFunc);
                 $('.modal.in').modal('hide');
                 $('.btn-group.open').removeClass('open');
              }
            };
            $('body').bind('keyup', keyupFunc);
        })
            .fail(function (msg) {
                $modal_plugin_log.find('.modal-body').html('<div class="log alert alert-error">Unable to read plugin log:' + htmlEscape(msg.statusText) + '</div>');
            })
            .always(function () {
                logParent.activity(false);
                $modal_plugin_log.modal('show');
            });
        //prevent the browser to follow the link
        return false;
    });


    /* Click handler for button to terminate running SGE job */
    $('.pluginCancel').live("click", function (e) {
        this.innerHTML = "Stopping";
        this.disabled = true;
        e.preventDefault();
        var prpk = $(this).data('id');
        var jobid = $(this).data('jobid');
        $.get("/rundb/api/v1/pluginresult/" + prpk + "/stop/"
            , function () {
                $('pluginStatusLoad').append('<div class="alert alert-info" data-dismiss="alert">Plugin Job ' + jobid + ' is being terminated via SGE</div>')
            }, 'json')
            .fail(function (msg) {
                $('pluginStatusLoad').append('<div class="alert alert-error" data-dismiss="alert">Failed to terminate SGE Job ' + jobid + '</div>')
            })
            .always(function () {
                // refresh plugin data
                setTimeout(pluginStatusLoad, 3000);
            });
    });

    $("#pluginRefresh").click(function () {
        pluginStatusLoad();
    });
    $("#pluginExpandAll").click(function () {
        $("#pluginStatusTable .plugin-collapse").text('-');
        $("#pluginStatusTable .pluginGroupList").slideDown('fast');
        var controls = {};
        for (var i = 0; i < table_plugin_names.length; i++)
            controls[table_plugin_names[i]] = true;
        update_plugin_show(controls);
    });
    $("#pluginCollapseAll").click(function () {
        $("#pluginStatusTable .plugin-collapse").text('+');
        $("#pluginStatusTable .pluginGroupList").slideUp('fast');
        var controls = {};
        for (var i = 0; i < table_plugin_names.length; i++)
            controls[table_plugin_names[i]] = false;
        update_plugin_show(controls);
    });

    //the plugin launcher
    $(".pluginDialogButton").click(function () {
        //open the dialog
        $("#modal-header").html('Select a plugin');
        $("#modal-body").html("<div id='pluginLoad'></div><div id='pluginList'></div>");
        $("#pluginLoad").html("<span>Loading Plugin List <img src='/site_media/jquery/colorbox/images/loading.gif'></img></span>");
        $('#plugin-modal').modal('show');

        //get the list of plugins from the API
        $.ajax({
            url: '/rundb/api/v1/plugin/?selected=true&limit=0&order_by=name',
            dataType: 'json',
            type: 'GET',
            async: false,
            success: function (data) {
                var items = [];
                if (data.objects.length === 0) {
                    $("#pluginLoad").html("");
                    $("#pluginList").html('<p>No plugins enabled. Go to <a href="/configure/plugins/">Configure:Plugins</a> to install and enable plugins.</p>');
                    return false;
                }
                $("#pluginList").html('<table id="plugin_table" class="table table-striped"></table>');
                plugins = data.objects.sort(function (a, b) {
                    return a.name.toLowerCase() > b.name.toLowerCase() ? 1 : -1;
                });
                for (var i = 0; i < plugins.length; i++) {
                    val = plugins[i];
                    if (val.isInstance) {
                        // Plugin needs parameters, popup instance config page
                        $("#plugin_table").append(
                            '<tr><td><a href="' + val.input +
                                '?report=' + djangoPK +
                                '"data-barcodes_table= "' + 'plugin/' + val.id + '/plugin_barcodes_table' +
                                '" class="plugin_input_class plugin_link colorinput" data-id="' + val.id +
                                '" id="' + val.name +
                                '_plugin">' + val.name + '</a> <small>&#8212; v' + val.version + '</small></td></tr>'
                        );
                    }
                    else {
                        // Plugin has no instance config, just run it.
                        $("#plugin_table").append(
                            '<tr><td><a href="#PluginOutput" class="plugin_class" data-id="' + val.id +
                                '" id="' + val.name +
                                '_plugin">' + val.name +
                                '</a> <small>&#8212; v' + val.version +
                                '</small></td></tr>'
                        );
                    }
                }
                $("#pluginLoad").html("");
            }
        });

        // We used to use the colorbox jquery plugin. Now we populate the bootstrap modal instead.
        (function () {
            var selectionModal = $("#plugin-modal");
            var iframeModal = $("#plugin-iframe-modal");
            var iframe = iframeModal.find("iframe");
            iframeModal.on('hide', function () {
                 iframe.attr("src", "");
            });
            // This jquery pluign resizes the frame after it loads a new url
            enableIframeResizing(iframe);
            $(".plugin_link").click(function (event) {
                event.preventDefault();
                iframe.css("height", 0);
                var iframe_src = $(this).attr("href");
                // load barcodes table first, before the plugin instance.html
                var barcodes_table_url = $(this).data('barcodes_table');
                iframeModal.find("#plugin_barcodes_table").load(barcodes_table_url, function(){
                    iframe.attr("src", iframe_src);
                });
                // Hide the selection modal and show the iframe vm
                selectionModal.modal("hide");
                iframeModal.modal('show');
                return false
            });
        })();

        //now the the for each is done, show the list
        $(".plugin_input_class").die("click");
        $(".plugin_input_class").live("click", function () {
            $('#plugin-modal').modal('hide');
            //Prevent background page scrolling
            document.body.style.overflow = 'hidden';

        });
        $(".plugin_class").die("click");
        $(".plugin_class").live('click', function () {
            //get the plugin id
            var id = $(this).data('id');
            var pluginName = $(this).attr("id");
            //get the plugin name
            pluginName = pluginName.substring(0, pluginName.length - 7);
            //build the JSON to post
            pluginAPIJSON = {
                //"result": "/rundb/api/v1/results/" + djangoPK + "/",
                "plugin": { "id": id, "name": pluginName },
                "pluginconfig": {}
            };
            //"plugin": { "id": id, "name": pluginName },
            //"pluginresult": pr_id, // rerun only
            pluginAPIJSON = JSON.stringify(pluginAPIJSON);
            $.ajax({
                type: 'POST',
                url: pluginURL,
                async: true,
                contentType: "application/json; charset=utf-8",
                data: pluginAPIJSON,
                dataType: "json",
                processData: false,
                beforeSend: function () {
                    $("#pluginList").html("");
                    $("#pluginLoad").html("<span>Launching Plugin " + pluginName + " <img src='/site_media/jquery/colorbox/images/loading.gif'></img></span>");
                },
                success: function () {
                    $('#plugin-modal').modal('hide');
                    pluginStatusLoad();
                },
                failure: function () {
                    $("#pluginLoad").html("<span>ERROR: Failed to launch Plugin " + pluginName + "</span>");
                }
            });
        });
    });

    //if this is a print view, unroll the tabs, and reformat the page as needed
    if ($.QueryString.no_header) {
        $(".tab-pane").each(function () {
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
            "margin": "20px"
        });
        $(".pluginGroup").css({
            "border": "1px black solid",
            "margin": "10px"
        });

        $(".unaligned .well").css("min-height", "390px");
        $(".aligned .well").css({"min-height": "370px", "height": "370px"});

        //remove the shadows around the content
        $(".content").css({"box-shadow": "0px 0px",
            "border-radius": "0px",
            "border": "0px"
        });

    } else {

        //Only do these things for the non print view

        //try to resize the plugin block all the time
        //Once a second is Plenty for this.

        resizeIntervalTimer = setInterval(function () {
            resizeiFrames();
        }, 1000);

        //init the Kendo Grids
        if (typeof CA_barcodes_json !== 'undefined') {
            $("#CA_barcodes").kendoGrid({
                dataSource: {
                    data: CA_barcodes_json,
                    schema: {
                        model: {
                            fields: {
                                ID: { type: "string" },
                                Filtered_Mapped_Bases_in_Q7_Alignments: { type: "integer" },
                                Total_number_of_Reads: { type: "integer" },
                                Filtered_Q7_Mean_Alignment_Length: { type: "integer" }
                            }
                        }
                    },
                    pageSize: 10
                },
                height: 'auto',
                groupable: false,
                scrollable: false,
                selectable: false,
                sortable: {
                    mode: "multiple",
                    allowUnsort: true
                },
                pageable: {pageSizes: [5, 10, 20, 50, 100, 1000]},
                columns: [
                    {
                        field: "ID",
                        title: "Barcode Name"
                    },
                    {
                        field: "Filtered_Mapped_Bases_in_Q7_Alignments",
                        title: "Aligned Output"
                    },
                    {
                        field: "Total_number_of_Reads",
                        title: "Reads"
                    },
                    {
                        field: "Filtered_Q7_Mean_Alignment_Length",
                        title: "Mean Aligned Read Length"
                    },
                    {
                        title: "BAM",
                        sortable: false
                    }
                ],
                rowTemplate: kendo.template($("#CA_barcodesRowTemplate").html())
            });
        }

        if (typeof barcodes_json !== 'undefined') {
            $("#barcodes").kendoGrid({
                dataSource: {
                    data: barcodes_json,
                    schema: {
                        model: {
                            fields: {
                                barcode_name: { type: "string" },
                                sample: { type: "string" },
                                total_bases: { type: "integer" },
                                Q20_bases: { type: "integer" },
                                read_count: { type: "integer" },
                                mean_read_length: { type: "string" },
                                filtered: { type: "boolean" },
                                file_prefix: { type: "string"},
                                bam_link: { type: "string"},
                                bai_link: { type: "string"},
                                basecaller_bam_link: { type: "string"}
                            }
                        }
                    },
                    pageSize: 10
                },
                height: 'auto',
                groupable: false,
                scrollable: false,
                selectable: false,
                sortable: {
                    mode: "multiple",
                    allowUnsort: true
                },
                pageable: {pageSizes: [5, 10, 20, 50, 100, 1000]},
                columns: [
                    {
                        field: "barcode_name",
                        title: "Barcode Name"
                    },
                    {
                        field: "sample",
                        title: "Sample"
                    },
                    {
                        field: "total_bases",
                        title: "Bases"
                    },
                    {
                        field: "Q20_bases",
                        title: ">=Q20 Bases"
                    },
                    {
                        field: "read_count",
                        title: "Reads"
                    },
                    {
                        field: "mean_read_length",
                        title: "Mean Read Length"
                    },
                    {
                        title: "Read Length Histogram",
                        sortable: false
                    },
                    {
                        title: "BAM",
                        sortable: false
                    }
                ],
                rowTemplate: kendo.template($("#barcodesRowTemplate").html())
            });
        }
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
        $("#calibration_report").kendoGrid({
            dataSource: {
                pageSize: 10
            },
            height: 'auto',
            groupable: false,
            scrollable: false,
            selectable: false,
            sortable: true,
            pageable: true
        });
    }
});
