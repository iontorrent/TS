(function ($) {
  // register namespace
  $.extend(true, window, {
    "Slick": {
      "CheckboxSelectColumn": CheckboxSelectColumn
    }
  });

  function CheckboxSelectColumn(options) {
    var _grid;
    var _self = this;
    var _selectFilterOn = false;
    var _origCell = 0
    var _origRow = 0;
    var _ignoreRowSel = false;
    var _clickSelect = false;
    var _filterCallback = null;
    var _defaults = {
      columnId: "_CBSEL",
      cssClass: null,
      toolTipOn: "Deselect All",
      toolTipOff: "Select All",
      toolTipDisabled: "Selected",
      checkColumnOn: "<img src='lifegrid/images/checkbox_tick.png'>",
      checkColumnOff: "<img src='lifegrid/images/checkbox_empty.png'>",
      checkColumnDisabled: "",
      width: 28
    };

    var _options = $.extend(true, {}, _defaults, options);

    function init(grid) {
      _grid = grid;
      _grid.onActiveCellChanged.subscribe(handleCellChanged);
      _grid.onSelectedRowsChanged.subscribe(handleSelectedRowsChanged);
      _grid.onClick.subscribe(handleClick);
      _grid.onHeaderClick.subscribe(handleHeaderClick);
      _grid.onKeyDown.subscribe(handleKeyDown);
    }

    function destroy() {
      _grid.onActiveCellChanged.unsubscribe(handleCellChanged);
      _grid.onSelectedRowsChanged.unsubscribe(handleSelectedRowsChanged);
      _grid.onClick.unsubscribe(handleClick);
      _grid.onHeaderClick.unsubscribe(handleHeaderClick);
      _grid.onKeyDown.unsubscribe(handleKeyDown);
    }

    function getColumnDefinition() {
      return {
        id: _options.columnId,
        name: _options.checkColumnOff,
        toolTip: _options.toolTipOff,
        field: "check",
        width: _options.width,
        resizable: false,
        sortable: false,
        cssClass: _options.cssClass,
        formatter: checkboxSelectionFormatter
      };
    }

    function setFilterSelected( value ) {
      if( value ) {
        _grid.updateColumnHeader(_options.columnId, _options.checkColumnDisabled, _options.toolTipDisabled);
        _selectFilterOn = true;
      } else {
        _grid.updateColumnHeader(_options.columnId, _options.checkColumnOff, _options.toolTipOff);
        _selectFilterOn = false;
      }
    }

    function setUpdateFilter( filter ) {
      _filterCallback = filter;
    }

    function checkAllSelected() {
      if( _selectFilterOn ) return;
      var pageSize = _grid.getDataLength();
      for( var i = 0; i < pageSize; ++i ) {
        if( !_grid.getData().getItem(i)['check'] ) {
          _grid.updateColumnHeader(_options.columnId, _options.checkColumnOff, _options.toolTipOff);
          return;
        }
      }
      _grid.updateColumnHeader(_options.columnId, _options.checkColumnOn, _options.toolTipOn);
    }

    function handleHeaderClick(e,args) {
      if( _selectFilterOn ) return;
      if( args.column.id !== _options.columnId ) return;
      e.stopPropagation();
      e.stopImmediatePropagation();
      var setCheck = (args.column.name === _options.checkColumnOff);
      var numRows = _grid.getDataLength();
      for( var i = 0; i < numRows; ++i )
      {
        if( _grid.getData().getItem(i)['check'] != setCheck )
        {
          _grid.getData().getItem(i)['check'] = setCheck;
          _grid.invalidateRow(i);
        }
      }
      if( setCheck ) {
        _grid.updateColumnHeader(_options.columnId, _options.checkColumnOn, _options.toolTipOn);
      } else {
        _grid.updateColumnHeader(_options.columnId, _options.checkColumnOff, _options.toolTipOff);
      }
      _grid.render();
    }

    function handleClick(e,args) {
      // clicking on a row select checkbox
      _clickSelect = true;
      _origCell = _targCell = args.cell;
      _origRow = args.row;
      _grid.setSelectedRows([args.row]);
    }

    function handleCellChanged(e,args) {
      _targCell = args.cell;
    }

    function handleSelectedRowsChanged(e,args) {
      // ignore recursive calls
      if( _ignoreRowSel ) {
        _ignoreRowSel = false;
        return;
      }
      var selectedRows = args.rows;
      if( selectedRows.length == 0 ) return;
      // disallow single row selection unless preceded by user click - prevents reset after page change
      if( selectedRows.length == 1 && !_clickSelect ) return;
      _clickSelect = false;
      // last (single) row selected is always at end of list (do not unselect this one)
      var targRow = (_origRow < selectedRows[0]) ? selectedRows[selectedRows.length-2] : selectedRows[0];
      // correction for shift-downarrow selection
      if( selectedRows.length == 2 && _origRow == targRow ) {
        targRow = selectedRows[selectedRows.length-1];
      }
      // check if making selection should affect checked columns
      var columns = _grid.getColumns();
      var clickOnCheck = (columns[_origCell].id === _options.columnId && columns[_targCell].id === _options.columnId);
      if( clickOnCheck ) {
        // for more intuitive use, set the selection to the complement of the target check
        var setTarg = !(_grid.getData().getItem(targRow)['check']);
        for( var row, i = 0; i < selectedRows.length; ++i ) {
          row = selectedRows[i];
          _grid.getData().getItem(row)['check'] = setTarg;
          _grid.invalidateRow(row);
        }
      }
      // leave the last row selecton (either pre-last or first)
      if( selectedRows.length > 1 ) {
        _ignoreRowSel = true; // prevent recursion
        _grid.setSelectedRows([targRow]);
        _grid.setActiveCell(targRow,_targCell);
      }
      if( clickOnCheck ) {
        if( _selectFilterOn && _filterCallback ) {
          _filterCallback();
        } else {
          _grid.render(); 
          checkAllSelected();
        }
      }
      _origRow = targRow;
      _origCell = _targCell;
    }

    function handleKeyDown(e,args) {
      // handle a space key as a click when cell is selected
      if( e.which == 32 ) {
        e.stopPropagation();
        e.stopImmediatePropagation();
        handleClick(e,args);
      }
/* disabled due to paging issues (etc)
      // have un-shifted up/down keys move selection row
      else if( e.which == 38 ) {
        if( args.row > 0 ) {
          _origCell = args.cell;
          _grid.setSelectedRows([args.row-1]);
        }
      }
      else if( e.which == 40 ) {
        if( args.row < _grid.getDataLength()-1 ) {
          _origCell = args.cell;
          _grid.setSelectedRows([args.row+1]);
        }
      }
*/
    }

    function checkboxSelectionFormatter(row, cell, value, columnDef, dataContext) {
      return value ? _options.checkColumnOn : _options.checkColumnOff;
    }

    $.extend(this, {
      "init": init,
      "destroy": destroy,
      "getColumnDefinition": getColumnDefinition,
      "checkAllSelected": checkAllSelected,
      "setUpdateFilter": setUpdateFilter,
      "setFilterSelected": setFilterSelected
    });
  }
})(jQuery);
