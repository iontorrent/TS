// shared lifegrid functions, etc.

// --- Deferred calls for after page loaded, e.g. for loading tables from files
var postPageLoadMethods = [];

window.onload= function() {
  // create priority mapping
  var loadOrder = [];
  for( var i = 0; i < postPageLoadMethods.length; ++i ) {
    var pr = postPageLoadMethods[i]['priority'];
    var j = 0;
    for( ; j < loadOrder.length; ++j ) {
      if( postPageLoadMethods[loadOrder[j]]['priority'] > pr ) break;
    }
    loadOrder.splice(j,0,i);
  }
  for( var i = 0; i < loadOrder.length; ++i ) {
    var j = loadOrder[i];
    postPageLoadMethods[j]['callback']();
  }
}

// --- filtering helpers
function strNoMatch(str,pat) {
  if( pat == "" ) return false;
  // return false if anywhere match where pattern starts with *
  if( pat.charAt(0) == '*' ) return (str.indexOf(pat.substr(1)) < 0);
  // return false if exact match at start of str
  return (str.indexOf(pat) != 0);
}

function rangeNoMatch(pos,first,last) {
  if( first == 0 ) return (last == 0) ? false : (pos > last);
  if( last < first ) return (pos < first);
  return (pos < first || pos > last);
}

function rangeInMatch(pos,first,last) {
  if( first == 0 ) return (last == 0) ? false : (pos <= last);
  if( last < first ) return (pos >= first);
  return (pos >= first && pos <= last);
}

function rangeLess(pos,low) {
  if( low === "" ) low = 0;
  return (pos < low);
}

function rangeMore(pos,high) {
  if( high === "" ) high = 0;
  return (pos > high);
}

function selectAppendUnique(selectID,text,value) {
  var selObj = $(selectID);
  if( !selObj || selObj == undefined ) return false;
  var options = $("option",selObj);
  for( var i = 0; i < options.length; ++i ) {
    if( options[i].text == text ) return false;
  }
  selObj.append("<option value='"+value+"'>"+text+"</option>");
  return true;
}

function forceStringFloat(value) {
  value = value.replace(/[^.\d]/g,"");
  var i = value.indexOf('.');
  if( i >= 0 )
  {
    i = value.indexOf('.',i+1);
    if( i > 0 ) value = value.substr(0,i);
  }
  if( value == "" ) return "";
  return value;
}

// --- simple cell data formatters
function formatPercent(row, cell, value, columnDef, dataContext) {
  if (value == null || value === "") { return "N/A" }
  return value.toFixed(1)+"%";
}

function formatScientific(row, cell, value, columnDef, dataContext) {
  if (value == null || value === "") { return "N/A" }
  return value.toExponential(2);
}


