var MOUSEOVER_DELAY = 500;

function newtab(url,name) {
    if (!name) {
	name = "";
    }
    window.open(url + '?src=' + escape(window.location.href),name,
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

function toggleTr(trid){
    var ele = $("#" + trid + "_holder");
    ele.slideToggle();
    return;
}

function toggleAdvanced(trid) {
    var tr = $("#" + trid).slideToggle();
}

function clickExpand(id) {
    var ctx = $("#" + id);
    var ele = $(".icon_link",ctx);
    var link = ele.attr('href');
    var code = link.substr(link.indexOf(':') + 1);
    ele.trigger('click');
    eval(code);
}

function link(url) {
    window.location.href=url;
}

function star(){
    var pk = this.id.split("_")[1];
    var setstr = null;
    if(this.checked){
	setstr = "1";
    }else{setstr = "0";}
    $.get("/rundb/star/"+pk+"/"+setstr, null, null, "text");
}

function storage_option(){
    var pk = this.id.split('_')[0];
    var name = this.id.split('_')[1];
    var setstr = this.storage_options.value;
    var message = null;
    if(setstr == 'A'){
	message = 'Archive';
    }
    else if(setstr == 'D'){
	message = 'Delete';
    }
    else if(setstr == 'KI'){
	message = 'Keep';
    }
    alert("Experiment "+name+" will now be a candidate to - "+message)
    //$.get("/rundb/storage/"+pk+"/"+setstr, null, null, "text");
    var url = "/rundb/storage/" + pk + "/" + setstr;
    $.ajax({
      type: 'POST',
      url: url
    });
}

function storage_option(){
    var pk = this.id.split('_')[0];
    var name = this.id.split('_')[1];
    var setstr = this.storage_options.value;
    var message = null;
    if(setstr == 'A'){
        message = 'Archive';
    }
    else if(setstr == 'D'){
        message = 'Delete';
    }
    else if(setstr == 'KI'){
        message = 'Keep';
    }
    alert("Experiment "+name+" will now be a candidate to - "+message)
    //$.get("/rundb/storage/"+pk+"/"+setstr, null, null, "text");
    var url = "/rundb/storage/" + pk + "/" + setstr;
    $.ajax({
                type: 'POST',
                url: url
            });
}

function enablePlugin(){
    var pk = this.id.split("_")[1];
    var setstr = null;
    if(this.checked){
	setstr = "1";
    }else{setstr = "0";}
    $.get("/rundb/enableplugin/"+pk+"/"+setstr, null, null, "text");
}

function enableEmail(){
    var pk = this.id.split("_")[1];
    var setstr = null;
    if(this.checked){
	setstr = "1";
    }else{setstr = "0";}
    $.get("/rundb/enableemail/"+pk+"/"+setstr, null, null, "text");
}

function enableArchive(){
    var pk = this.id.split("_")[1];
    var setstr = null;
    if(this.checked){
	setstr = "1";
    }else{setstr = "0";}
    $.get("/rundb/enablearchive/"+pk+"/"+setstr, null, null, "text");
}

function enableTestFrag(){
    var pk = this.id.split("_")[1];
    var setstr = null;
    if(this.checked){
	setstr = "1";
    }else{setstr = "0"};
    $.get("/rundb/enabletestfrag/"+pk+"/"+setstr, null, null, "text");
}

function get_progress_bar(){
    var id = this.id;
    $.getJSON("progress_bar/" + id, null, set_pb_factory(id));
}

function set_pb_factory(id){
    return function(data){
        var value = data['value'];
        $(".progDiv").html();
        $("#" + id).progressbar('destroy');
        $("#" + id).progressbar({'value':value}).children('.ui-progressbar-value').html(value + '%');


    }
}

function get_progressbox(){
    var id = this.id;
    $.getJSON("progressbox/" + id, null, set_progressbox(id));
}

function set_progressbox(id){
    return function(data){
	var value = data['value'];
	$.each(value,function(key,value){
	    $('#'+id).find('#'+key).css('background-color', value);
	    });
	}
}

function do_control(url) {
    $.getJSON(url,function(data) {
	    var success = data[0];
	    var _status_d = $("#control_status_dialogue");
	    _status_d.dialog({bgiframe:true,
			height:240,
			width:480,
			resizeable:false,
			close:function(event,ui) {
			          window.location.reload(true);}
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

function build_control_dialogue(url,name) {

    var _control_d = $("#control_dialogue");
    _control_d.dialog({bgiframe:true,
		height:280,
		width:480,
		modal:true,
		resizable:false});
    var job_name = $("#control_dialogue > p > #job_name");
    job_name.text(name);
    $("#term_button_holder > input").click(function() {
	    _control_d.dialog('close');
	    do_control(url);
	});
    $("#cancel_button_holder > input").click(function() {
	    _control_d.dialog('close');
	});
    _control_d.dialog('open');
}


function do_delete(url) {

            $.post(url, function(data){
                var _status_d = $("#control_status_dialogue");
                _status_d.dialog({bgiframe:true,
                    height:280,
                    width:480,
                    resizeable:false,
                    close:function(event,ui) {
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
    _control_d.dialog({bgiframe:true,
        height:280,
        width:480,
        modal:true,
        resizable:false});
    $("#term_button_holder > input").click(function() {
        _control_d.dialog('close');
        do_delete(url);
    });
    $("#cancel_button_holder > input").click(function() {
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
    var width = parseInt(Math.min(800,(bw + 1.5)*hps.length + 100));
    var bg = graph.parent().css('background-color');
    var baropts = {
	data:data,
	size: String(width) + "x320",
	bar_width:2,
	title:"",
	legend: ["T", "A", "C", "G"],
	colors: ["ff6666", "66b366", "6666ff", "666666"],
	bg: "00000000",
	chxs:[[0,"000000"],[1,"000000"]],
	chd: [0,0,Math.max(hps)+1,1]
    };
    var img = $("<img />");
    var url = api.make(baropts) + escape("&chxs=0,000000,1,000000");
    img.attr('src', url).appendTo(graph);
    //graph.width(width);
    graph.css('margin-left','auto');
    graph.css('margin-right','auto');

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
	ret = $("div:first",sortable_th).eq(0).text().toLowerCase();
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
    var sel = ".sortables > th:contains(" + cap(ssf)  + ")";
    var ele = $(sel);
    if (ele.length == 0) {
	sel = ".sortables > th > .sortkey:contains(" + ssf + ")";
	ele = $(sel);
	var ret = null;
	try {
	    ret =  ele.get(0).parentNode;
	} catch(exn) {}
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

function sortClickCbFactory(rev,sortkey,sfelement) {
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
	icon.attr("title","Sort ascending");
	icon.click(sortClickCbFactory(false,key,sfelement));
	rev_icon.addClass("sort_rev_icon");
	rev_icon.addClass("ui-icon-arrowthick-1-s");
	rev_icon.click(sortClickCbFactory(true,key,sfelement));
	rev_icon.attr("title","Sort descending");
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
	    jqele.css("color","blue");
	}
	icon.insertAfter($(".sortheading",jqele));
	rev_icon.insertAfter($(".sortheading",jqele));
    }
}

function cap(s) {
    return s.charAt(0).toUpperCase() + s.substr(1).toLowerCase();
}

function docHasSortable() {
    var sf = $(".sortfield");
    return sf.length == 1;
}

/* TOOLTIPS */

var TOOLTIP_ELE = null;
var TOOLTIP_KEYS = {};

function tooltipCbFactory(ele,url) {
    var DIALOG_WIDTH = 320;
    var MARGIN = 20;
    function cb(data) {
	var d = null;
	if (TOOLTIP_ELE === null) {
	    TOOLTIP_ELE = true;
	    var rawdialog = $("<div />");
	    rawdialog.css("display","none");
	    d = $(rawdialog).dialog({
		    width:DIALOG_WIDTH,
		    height:"auto",
		    autoOpen:false,
		    resizable:false,
		    title:"Tooltip"
		});
	    TOOLTIP_ELE = d;
	} else {
	    d = TOOLTIP_ELE;
	}
	if (TOOLTIP_KEYS[url] == null) {
	    TOOLTIP_KEYS[url] = data;
	}
	if (typeof(data) == 'object') {
	    d.dialog('option', 'title', data['title']);
	    d.text(data['text']);
	} else {
	    d.dialog('option', 'title', '');
	    d.text(data);
	}
	off = ele.offset();
	var right = $(window).width() - (off.left + DIALOG_WIDTH + MARGIN);
	var hidden = Math.min(0, right);
	var left_pos = off.left + hidden;
	d.dialog('option','position',[left_pos,off.top + ele.height() + 2]);
	d.dialog('option','close',function(event,ui) {
		ele.removeClass("tooltip_highlighted");});
	d.dialog('open');
    }
    return cb;
}

HOVER_TIMEOUTS = {};


function tooltipClose() {
    var ele = $(this);
    var url = extractUrl(ele);
    ele.removeClass("tooltip_highlighted");
    var timeout = HOVER_TIMEOUTS[url];
    if (timeout) {
	clearTimeout(timeout);
    }
    if (TOOLTIP_ELE) {
	TOOLTIP_ELE.dialog('close');
    }
}

function extractUrl(ele) {
    return $(".tooltip",ele).text();
}

function _tt(url,cb) {
    var prev = TOOLTIP_KEYS[url];
    if (prev == null) {
	$.getJSON(url,null,cb);
    } else {
	return cb(prev);
    }
}

function retrieveTooltip(no_cb) {
    var ele = $(this);
    ele.addClass("tooltip_highlighted");
    var url = extractUrl(ele);
    function _cb() {
	var cb = tooltipCbFactory(ele,url);
	return _tt(url,cb);
    }
    var t = setTimeout(_cb,MOUSEOVER_DELAY);
    HOVER_TIMEOUTS[url] = t;
}

function displaySummary(d,keys) {
    var out = $("<div />");
    var ul = $("<ul />");
    function blink(k) {
	function cb() {
	    var ref = $(".tooltip:contains(" + k + ")").parent();
	    var i = 0;
	    for (i = 0; i < 2; i++) {
		ref.animate({opacity:0.0},100)
		    .animate({opacity:1.0},150);
	    }
	}
	return cb;
    }
    function addObj(li,data,key) {
	var h4 = $("<h4 />");
	h4.click(blink(key));
	h4.css("cursor","pointer");
	h4.css("margin","2px");
	var span = $("<span />");
	span.css("border-bottom","1px dotted black");
	span.text(data["title"]);
	h4.append(span);
	li.append(h4);
	var body = $("<div />");
	body.text(data["text"]);
	li.append(body);
    }
    function addStr(li,data,key) {
	var body = $("<div />");
	body.text(data);
	body.click(blink(key));
	body.css("cursor", "pointer");
	body.css("border-bottom", "1px dotted black");
	li.append(body);
    }
    for (ndx in keys) {
	var k = keys[ndx];
	var v = d[k];
	var li = $("<li />");
	if (typeof(v) == "object") {
	    addObj(li,v,k);
	} else {
	    addStr(li,v,k);
	}
	li.css("margin-bottom","8px");
	ul.append(li);
    }
    out.append(ul);
    out.dialog({width:800,height:"auto",title:"Help",modal:true});
    out.dialog('open');
}

function retrieveAllTooltips(tips) {
    var d = {};
    var count = 0;
    function cbFactory(url) {
	function cb(data) {
	    d[url] = data;
	    count++;
	    if (count == tips.length) {
		displaySummary(d,keys);
	    }
	}
	return cb;
    }
    var ndx;
    var keys = [];
    for (ndx=0; ndx < tips.length; ndx++) {
	var t = tips.eq(ndx);
	var url = extractUrl(t);
	keys.push(url);
    }
    for (ndx in keys) {
	var k = keys[ndx];
    	_tt(k,cbFactory(k));
    }
}

function gatherTooltips() {
    retrieveAllTooltips($(".tooltip").parent());
}

/* END TOOLTIPS */

/* DOCUMENT READY CALLBACKS */

/* TEMPLATE INITIALIZERS */
function prep_graphable() {
    $(".graphable").each(function(){
	    var id = this.id;
	    var pk = id.split('_')[0];
	    var seq = $("#" + id + " > .sequence").text().toLowerCase();
	    var graph = $("#" + pk + "_graph");
	    graph_template(seq, graph);
	});
}

/* TOOLTIP INITIALIZERS */
function prep_tooltip() {
    var ps = $(".tooltip").parent();
    if (ps.length > 0) {
	ps.addClass("tooltip_parent");
	ps.hover(retrieveTooltip,tooltipClose);
    }
}
function prep_tooltip_summary() {
    $(".tooltip_summary").click(gatherTooltips);
}

/* CONTROL FORM INITIALIZERS */
function prep_controlform() {
    var domstr = "#control_form > table > tbody > tr > td > ";
    $(domstr + "input").change(submitControlForm);
    $(domstr + "select").change(submitControlForm);
}

/* TAB EFFECTS INITIALIZERS */
function prep_tabs() {
    $(".tabtext").hover(function(){
	    var ele = $(this);
	    if (!ele.hasClass("selected")) {
		ele.css("background-color","#ffffff");
	    }},
	function(){
	    var ele = $(this);
	    if (!ele.hasClass("selected")) {
		ele.css("background-color","#cccccc");
	    }});
}
function prep_tab_corners() {
    var tablist = $(".tabtext");
    tablist.addClass("ui-corner-tl");
    tablist.addClass("ui-corner-tr");
}

/* PLUGIN INITIALIZERS */
function prep_enable_plugin() {
    $(".enable_plugin_td > input").change(enablePlugin);
}

//function prep_enable_plugin_autorun() {
//    $(".enable_plugin_autorun_td > input").change(enablePlugin);
//}

/* TEST FRAGMENT INITIALIZER */
function prep_enable_test_fragment() {
    $(".testfrag_td > input").change(enableTestFrag);
}

/* EMAIL INITIALIZERS */
function prep_enable_email() {
    $(".enable_email_td > input").change(enableEmail);
}

/* Archive INITIALIZERS */
function prep_enable_archive() {
    $(".archive_td > input").change(enableArchive);
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
    $('.progress_bar').progressbar({value:0});
}

/* PROGRESS BOX INITIALIZER */
function prep_progressbox() {
    $('.progressbox').get_progressbox();
}

/* ICON INITIALIZERS */
function prep_icon_toggling() {
    function cb() {
	var ele = $(this);
	var icon1 = $(".__icon_1",ele).text();
	var icon2 = $(".__icon_2",ele).text();
	if (icon2.length > 0 && icon1.length > 0) {
	    var toset = $(this.firstChild);
	    toset.toggleClass(icon1);
	    toset.toggleClass(icon2);
	}
    }
    $(".icon_link:has(.__icon_2)").click(cb);
}
function prep_icon_effects() {
    $(".icon_link").hover(function() {
	    var child = $(this.firstChild);
	    if (!child.hasClass("ui-state-disabled")) {
		$(this).addClass("ui-state-highlight");
	    }},
	function() {
	    $(this).removeClass("ui-state-highlight");
	});
}

/* SORTING INITIALIZERS */
function prep_sorting_text() {
    $(".sortables > th > .sortheading").click(function() {
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
    mw = parseInt(mw.substr(0,mw.length - 2)) - 20;
    function cb() {
	var docw = $(window).width();
	var outw = Math.max(mw,docw-20);
	all.width(outw);
    }
    $(window).resize(cb);
    all.removeClass("all_width");
    cb();
}
function prep_trim_text() {
    $(".trimtext").each(function() {
	    var t = $(this).text();
	    var lim = 18;
	    if (t.length > lim) {
		$(this).text(t.substr(0,lim-3) + '...');
		$(this).attr("title",t);
	    }
	});
}

//Functions/methdos added by Nidhi Tare. These methods are usable in 1_Torrent_Accuracy Plugin and are planned to be introduced in the future plugins as well

function goBack() {
//Function that creates a back button on the plugin report
window.history.back()
}


function toggle(myid, alt, target){
//Function that creates the Expand and Collapse buttons in the Report. Users expands to view a detailed report and when he needs to hide the detailed report he clicks View Less button. Tested to work with both firefox and Chrome
// TO DO: Write a function with fewer function arguments.

    var hideReport = document.getElementById(myid);
    var toggle = document.getElementById(alt);
    //var target = event.target.id; Does not work with Firefox
    
    if (hideReport.style.display == '') //When detailed report is unhidden and View Less is Clicked
    {
        hideReport.style.display = 'none';
        toggle.innerHTML = "View More...";
        document.getElementById(target).innerHTML='';
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
$(document).ready(prep_tooltip);
$(document).ready(prep_tooltip_summary);
$(document).ready(prep_graphable);
$(document).ready(prep_controlform);
$(document).ready(prep_tabs);
$(document).ready(prep_star);
$(document).ready(prep_storage_option);
$(document).ready(prep_progress_bar);
$(document).ready(function() {
	setInterval(function() {
		$(".progress_bar").each(get_progress_bar)},60000)});
$(document).ready(function() {
		    $(".progress_bar").each(get_progress_bar)});
$(document).ready(prep_icon_toggling);
$(document).ready(prep_tab_corners);
$(document).ready(prep_sorting_text);
$(document).ready(prep_sorting_selected);
$(document).ready(prep_sorting_buttons);
$(document).ready(prep_centering_ie6);
$(document).ready(prep_centering_width);
$(document).ready(prep_icon_effects);
$(document).ready(prep_trim_text);
$(document).ready(prep_enable_plugin);
//$(document).ready(prep_enable_plugin_autorun);
$(document).ready(prep_enable_email);
$(document).ready(prep_enable_test_fragment);
$(document).ready(prep_enable_archive);
//$(document).ready(prep_progressbox);
$(document).ready(function() {
    setInterval(function() {
		    $(".progressbox_holder").each(get_progressbox)},60000)});
$(document).ready(function() {
		    $(".progressbox_holder").each(get_progressbox)});
