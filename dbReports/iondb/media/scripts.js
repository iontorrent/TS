var MOUSEOVER_DELAY = 500;

function newtab(url, name) {
    if (!name) {
        name = "";
    }
    window.open(url + '?src=' + escape(window.location.href), name,
        "menubar=no,width=480,height=360,toolbar=no");
}

function closetab() {
    self.close();
}

/*
 function toggleDiv(divid){
 if(document.getElementById(divid).style.display == 'none'){
 document.getElementById(divid).style.display = 'block';
 }else{
 document.getElementById(divid).style.display = 'none';
 }
 }*/

function toggleTr(trid) {
    var ele = $("#" + trid + "_holder");
    ele.slideToggle();
    return;
}

function toggleAdvanced(trid) {
    var tr = $("#" + trid).slideToggle();
}

function clickExpand(id) {
    var ctx = $("#" + id);
    var ele = $(".icon_link", ctx);
    var link = ele.attr('href');
    var code = link.substr(link.indexOf(':') + 1);
    ele.trigger('click');
    eval(code);
}

function link(url) {
    window.location.href = url;
}

function star() {
    var pk = this.id.split("_")[1];
    var setstr = null;
    if (this.checked) {
        setstr = "1";
    } else {
        setstr = "0";
    }
    $.get("/rundb/star/" + pk + "/" + setstr, null, null, "text");
}

function storage_option() {
    var pk = this.id.split('_')[0];
    var name = this.id.split('_')[1];
    var setstr = this.storage_options.value;
    var message = null;
    if (setstr == 'A') {
        message = 'Archive';
    }
    else if (setstr == 'D') {
        message = 'Delete';
    }
    else if (setstr == 'KI') {
        message = 'Keep';
    }
    alert("Experiment " + name + " will now be a candidate to - " + message)
    //$.get("/rundb/storage/"+pk+"/"+setstr, null, null, "text");
    var url = "/rundb/storage/" + pk + "/" + setstr;
    $.ajax({
        type: 'POST',
        url: url
    });
}

function storage_option() {
    var pk = this.id.split('_')[0];
    var name = this.id.split('_')[1];
    var setstr = this.storage_options.value;
    var message = null;
    if (setstr == 'A') {
        message = 'Archive';
    }
    else if (setstr == 'D') {
        message = 'Delete';
    }
    else if (setstr == 'KI') {
        message = 'Keep';
    }
    alert("Experiment " + name + " will now be a candidate to - " + message)
    //$.get("/rundb/storage/"+pk+"/"+setstr, null, null, "text");
    var url = "/rundb/storage/" + pk + "/" + setstr;
    $.ajax({
        type: 'POST',
        url: url
    });
}

function report_option() {
    var pk = this.id.split('_')[0];
    var len1 = this.id.split('_')[0].length;
    var name = this.id.substring(len1+1,this.id.length);
    var setstr = this.report_options.value;
    var message = null;
    if (setstr == 'E') {
	message = 'Export';
    }
    else if (setstr == 'A') {
	message = 'Archive';
    }
    else if (setstr == 'D') {
	message = 'Delete';
    }
    else if (setstr == 'P') {
	message = 'Prune';
    }
    else if (setstr == 'Z') {
	message = 'switch exempt status';
    }
    if (setstr != 'N') {
	var url = "/report/action/" + pk + "/" + setstr;
	var r = prompt("Report " + name + " will now " + message + ". Proceed?\nUpdate comment:", "");
	if(r!=null)
	{
		var data = {};
		data.comment = r;
		$.ajax({
		    type: 'POST',
		    data: data,
		    url: url,
		    success: function(){window.location.reload(true);},
		    error: function(){window.location.reload(true);}
		});
	}
	else
	{
		location.reload(true);
	}
    }
    else {
	alert("Please select an option to perform an action on report " + name + ".")
    }
}

function get_progress_bar() {
    var id = this.id;
    $.getJSON("progress_bar/" + id, null, set_pb_factory(id));
}

function set_pb_factory(id) {
    return function (data) {
        var value = data['value'];
        $(".progDiv").html();
        $("#" + id).progressbar('destroy');
        $("#" + id).progressbar({'value': value}).children('.ui-progressbar-value').html('<div class="progress-number">' + value + '%</div>');
    }
}

function get_progressbox() {
    var id = this.id;
    //stagger the requests by 3 seconds
    var randoTime= Math.floor((Math.random()*3000)+1);
    setTimeout(function(){
        $.getJSON("progressbox/" + id, null, set_progressbox(id));
    }, randoTime);
}

function set_progressbox(id) {

    return function (data) {
        //need to update the tooltips too
        var tooltip = {
            "completed":"The run analysis has completed",
            "error":"The run analysis failed, Please check run log for specific error",
            "terminated":"User terminated analysis job",
            "started": "The analysis is currently processing",
            "checksum":"One of the raw signal files (DAT) is corrupt. Try re-transferring the data from the PGM",
            "pgm operation error": "Unexpected raw data values. Please check PGM for clogs or problems with chip"
        };
        var value = data['value'];
        $.each(value, function (key, value) {
            $('#' + id).find('#' + key).css('background-color', value);
        });
        var status = data['status'];
        $('#' + id).parent().parent().find(".hasTip").text(status);
        $('#' + id).parent().parent().find(".hasTip").attr("title", tooltip[status.toLowerCase()]);
        $('.hasTip').tipTip({ position: 'bottom' });

    }
}

function do_control(url) {
    $.getJSON(url, function (data) {
        var success = data[0];
        var _status_d = $("#control_status_dialogue");
        _status_d.dialog({bgiframe: true,
            height: 240,
            width: 480,
            resizeable: false,
            close: function (event, ui) {
                window.location.reload(true);
            }
        });
        var msgarea = $("#job_status_text");
        if (success) {
            msgarea.text("Termination succeeded.");
        } else {
            msgarea.text("Termination failed.");
        }
        _status_d.dialog("open");
    });
}

function build_control_dialogue(url, name) {

    var _control_d = $("#control_dialogue");
    _control_d.dialog({bgiframe: true,
        height: 280,
        width: 480,
        modal: true,
        resizable: false});
    var job_name = $("#control_dialogue > p > #job_name");
    job_name.text(name);
    $("#term_button_holder > input").click(function () {
        _control_d.dialog('close');
        do_control(url);
    });
    $("#cancel_button_holder > input").click(function () {
        _control_d.dialog('close');
    });
    _control_d.dialog('open');
}


function do_delete(url) {

    $.post(url, function (data) {
        var _status_d = $("#control_status_dialogue");
        _status_d.dialog({bgiframe: true,
            height: 280,
            width: 480,
            resizeable: false,
            close: function (event, ui) {
                self.location = "/rundb/references/";
            }
        });
        var msgarea = $("#job_status_text");

        if (data) {
            msgarea.html(data.status);
        }
        _status_d.dialog("open");
    });

}
function build_genome_dialogue(url) {

    var _control_d = $("#control_dialogue");
    _control_d.dialog({bgiframe: true,
        height: 280,
        width: 480,
        modal: true,
        resizable: false});
    $("#term_button_holder > input").click(function () {
        _control_d.dialog('close');
        do_delete(url);
    });
    $("#cancel_button_holder > input").click(function () {
        _control_d.dialog('close');
    });
    _control_d.dialog('open');
}

function graph_template(seq, graph) {
    if (seq.length <= 0) {
        return;
    }
    var hps = [];
    var flows = 'tacg';
    var ndx = 0;
    while (ndx < seq.length) {
        var prevndx = ndx;
        for (var b = 0; b < flows.length; b++) {
            var hp_len = 0;
            var base = flows.charAt(b);
            while (true) {
                var c = seq.charAt(ndx);
                if (c != base) {
                    break;
                } else if (flows.indexOf(c) == -1) {
                    break;
                }
                if (ndx >= seq.length) {
                    break;
                }
                ndx++;
                hp_len++;
            }
            hps.push(hp_len);
        }
        if (prevndx == ndx) {
            break;
        }
    }
    var api = jGCharts.Api();
    var data = [];
    var curr = [];
    for (i = 0; i < hps.length; i++) {
        if (!(i % flows.length)) {
            if (curr.length) {
                data.push(curr);
            }
            curr = [];
        }
        curr.push(hps[i]);
    }
    var bw = 3;
    var width = parseInt(Math.min(800, (bw + 1.5) * hps.length + 100));
    var bg = graph.parent().css('background-color');
    var baropts = {
        data: data,
        size: String(width) + "x320",
        bar_width: 2,
        title: "",
        legend: ["T", "A", "C", "G"],
        colors: ["ff6666", "66b366", "6666ff", "666666"],
        bg: "00000000",
        chxs: [
            [0, "000000"],
            [1, "000000"]
        ],
        chd: [0, 0, Math.max(hps) + 1, 1]
    };
    var img = $("<img />");
    var url = api.make(baropts) + escape("&chxs=0,000000,1,000000");
    img.attr('src', url).appendTo(graph);
    //graph.width(width);
    graph.css('margin-left', 'auto');
    graph.css('margin-right', 'auto');

}

function submitControlForm() {
    var form = $("#control_form");
    var ele = form.get(0);
    if (ele !== null && ele.submit.click) {
        ele.submit.click();
    }
}

/* SORTING FUNCTIONS */
/*
 function sortBase(sortfield) {
 if (sortfield.charAt(0) == '-') {
 base = sortfield.substr(1);
 rev = true;
 } else {
 base = sortfield;
 rev = false;
 }
 return [base,rev];
 }*/

function setSorting(sortkey) {
    var field = $(".sortfield");
    if (sorterIsSelected(sortkey)) {
        var rev = selectedSortIsReversed();
        if (!rev) {
            sortkey = '-' + sortkey;
        }
    }
    field.val(sortkey);
    submitControlForm();
}

function getSortKey(sortable_th) {
    var sortkeydiv = $(".sortkey", sortable_th);
    if (sortkeydiv.length > 0) {
        ret = sortkeydiv.text().toLowerCase();
    } else {
        ret = $("div:first", sortable_th).eq(0).text().toLowerCase();
    }
    return ret;
}

/*
 function getSortIcon(ele,rev) {
 sel = "";
 if (rev) {
 full = sel + ".sort_rev_icon";
 } else {
 full = sel + ".sort_icon";
 }
 return $(full,ele);
 }*/

function selectedSortField() {
    var ret = $(".sortfield").val().toLowerCase();
    if (ret.charAt(0) == '-') {
        return ret.substr(1);
    } else {
        return ret;
    }
}

function selectedSortTh() {
    var ssf = selectedSortField();
    var sel = ".sortables > th:contains(" + cap(ssf) + ")";
    var ele = $(sel);
    if (ele.length == 0) {
        sel = ".sortables > th > .sortkey:contains(" + ssf + ")";
        ele = $(sel);
        var ret = null;
        try {
            ret = ele.get(0).parentNode;
        } catch (exn) {
        }
        return ret;
    }
    return ele.get(0);
}

function selectedSortIsReversed() {
    return $(".sortfield").val().charAt(0) == '-';
}

function sorterIsSelected(sortkey) {
    return sortkey == selectedSortField();
}

function sortClickCbFactory(rev, sortkey, sfelement) {
    function cb() {
        if (rev) {
            sortkey = '-' + sortkey;
        }
        sfelement.val(sortkey);
        submitControlForm();
    }

    return cb
}

function addSortIcon() {
    var jqele = $(this);
    if (jqele.text()) {
        var icon = $("<span />");
        var rev_icon = $("<span />");
        var key = getSortKey(jqele);
        var sfelement = $(".sortfield").eq(0);
        var clses = ["ui-icon", "ui-corner-all", "icon_link",
            "ui-state-default"];
        var ndx = null;
        for (ndx in clses) {
            var curr = clses[ndx];
            icon.addClass(curr);
            rev_icon.addClass(curr);
        }
        icon.addClass("sort_icon");
        icon.addClass("ui-icon-arrowthick-1-n");
        icon.attr("title", "Sort ascending");
        icon.click(sortClickCbFactory(false, key, sfelement));
        rev_icon.addClass("sort_rev_icon");
        rev_icon.addClass("ui-icon-arrowthick-1-s");
        rev_icon.click(sortClickCbFactory(true, key, sfelement));
        rev_icon.attr("title", "Sort descending");
        if (sorterIsSelected(key)) {
            var sel_icon = null;
            if (selectedSortIsReversed()) {
                sel_icon = rev_icon;
            } else {
                sel_icon = icon;
            }
            sel_icon.unbind('click');
            sel_icon.unbind('hover');
            sel_icon.addClass("ui-state-active");
            jqele.css("color", "blue");
        }
        icon.insertAfter($(".sortheading", jqele));
        rev_icon.insertAfter($(".sortheading", jqele));
    }
}

function cap(s) {
    return s.charAt(0).toUpperCase() + s.substr(1).toLowerCase();
}

function docHasSortable() {
    var sf = $(".sortfield");
    return sf.length == 1;
}

/* DOCUMENT READY CALLBACKS */

/* TEMPLATE INITIALIZERS */
function prep_graphable() {
    $(".graphable").each(function () {
        var id = this.id;
        var pk = id.split('_')[0];
        var seq = $("#" + id + " > .sequence").text().toLowerCase();
        var graph = $("#" + pk + "_graph");
        graph_template(seq, graph);
    });
}

/* CONTROL FORM INITIALIZERS */
function prep_controlform() {
    var domstr = "#control_form > table > tbody > tr > td > ";
    $(domstr + "input").change(submitControlForm);
    $(domstr + "select").change(submitControlForm);
}

//REPORT ACTION OPTION INITIALIZER
function prep_report_option() {
    $(".report_td > form").change(report_option);
}

/* TAB EFFECTS INITIALIZERS */
function prep_tabs() {
    $(".tabtext").hover(function () {
            var ele = $(this);
            if (!ele.hasClass("selected")) {
                ele.css("background-color", "#ffffff");
            }
        },
        function () {
            var ele = $(this);
            if (!ele.hasClass("selected")) {
                ele.css("background-color", "#cccccc");
            }
        });
}
function prep_tab_corners() {
    var tablist = $(".tabtext");
    tablist.addClass("ui-corner-tl");
    tablist.addClass("ui-corner-tr");
}

/* STAR INITIALIZERS */
function prep_star() {
    $(".star_td > input").change(star);
}

//STORAGE OPTION INITILIZER
function prep_storage_option() {
    $(".storage_td > form").change(storage_option);
}

/* PROGRESS BAR INITIALIZER */
function prep_progress_bar() {
    $('.progress_bar').progressbar({value: 0});
}

/* PROGRESS BOX INITIALIZER */
function prep_progressbox() {
    $('.progressbox').get_progressbox();
}

/* ICON INITIALIZERS */
function prep_icon_toggling() {
    function cb() {
        var ele = $(this);
        var icon1 = $(".__icon_1", ele).text();
        var icon2 = $(".__icon_2", ele).text();
        if (icon2.length > 0 && icon1.length > 0) {
            var toset = $(this.firstChild);
            toset.toggleClass(icon1);
            toset.toggleClass(icon2);
        }
    }

    $(".icon_link:has(.__icon_2)").click(cb);
}
function prep_icon_effects() {
    $(".icon_link").hover(function () {
            var child = $(this.firstChild);
            if (!child.hasClass("ui-state-disabled")) {
                $(this).addClass("ui-state-highlight");
            }
        },
        function () {
            $(this).removeClass("ui-state-highlight");
        });
}

/* SORTING INITIALIZERS */
function prep_sorting_text() {
    $(".sortables > th > .sortheading").click(function () {
        setSorting(getSortKey($(this).parent()));
    });
}

function prep_sorting_selected() {
    if (docHasSortable()) {
        var sf = $(".sortfield");
        if (!sf.val()) {
            sf.val("-date");
        }
    }
}
function prep_sorting_buttons() {
    if (docHasSortable()) {
        var sortables = $(".sortables > th");
        sortables.each(addSortIcon);
    }
}

/* CENTERING INITIALIZERS */
function prep_centering_ie6() {
    if (!$.support.boxModel) {
        $(".centered").addClass("centered_ie6");
    }
}
function prep_centering_width() {
    var all = $(".all");
    var mw = all.css("min-width");
    mw = parseInt(mw.substr(0, mw.length - 2)) - 20;
    function cb() {
        var docw = $(window).width();
        var outw = Math.max(mw, docw - 20);
        all.width(outw);
    }

    $(window).resize(cb);
    all.removeClass("all_width");
    cb();
}
function prep_trim_text() {
    $(".trimtext").each(function () {
        var t = $(this).text();
        var lim = 18;
        if (t.length > lim) {
            $(this).text(t.substr(0, lim - 3) + '...');
            $(this).attr("title", t);
        }
    });
}

//Functions/methdos added by Nidhi Tare. These methods are usable in 1_Torrent_Accuracy Plugin and are planned to be introduced in the future plugins as well

function goBack() {
//Function that creates a back button on the plugin report
    window.history.back()
}


function toggle(myid, alt, target) {
//Function that creates the Expand and Collapse buttons in the Report. Users expands to view a detailed report and when he needs to hide the detailed report he clicks View Less button. Tested to work with both firefox and Chrome
// TO DO: Write a function with fewer function arguments.

    var hideReport = document.getElementById(myid);
    var toggle = document.getElementById(alt);
    //var target = event.target.id; Does not work with Firefox

    if (hideReport.style.display == '') //When detailed report is unhidden and View Less is Clicked
    {
        hideReport.style.display = 'none';
        toggle.innerHTML = "View More...";
        document.getElementById(target).innerHTML = '';
    }
    else //When View More is Clicked
    {
        hideReport.style.display = '';
        toggle.innerHTML = 'View Less...';
        document.getElementById(target).innerHTML = '';
    }


}


// Order of jquery callbacks. Callbacks that add styling should be
// last in the order, so that CSS classes added by previous callbacks
// are found by styling initializers.
$(document).ready(prep_graphable);
$(document).ready(prep_controlform);
$(document).ready(prep_tabs);
$(document).ready(prep_star);
$(document).ready(prep_storage_option);
$(document).ready(prep_report_option);
$(document).ready(prep_progress_bar);
$(document).ready(function () {
    setInterval(function () {
        $(".progress_bar").each(get_progress_bar)
    }, 60000)
});
$(document).ready(function () {
    $(".progress_bar").each(get_progress_bar)
});
$(document).ready(prep_icon_toggling);
$(document).ready(prep_tab_corners);
$(document).ready(prep_sorting_text);
$(document).ready(prep_sorting_selected);
$(document).ready(prep_sorting_buttons);
$(document).ready(prep_centering_ie6);
$(document).ready(prep_centering_width);
$(document).ready(prep_icon_effects);
$(document).ready(prep_trim_text);
//$(document).ready(prep_progressbox);
$(document).ready(function () {
    setInterval(function () {
        $(".progressbox_holder").each(get_progressbox)
    }, 45000)
});
$(document).ready(function () {
    $(".progressbox_holder").each(get_progressbox)
});
