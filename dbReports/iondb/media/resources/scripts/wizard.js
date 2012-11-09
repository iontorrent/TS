function setTab(tabId) {
	// alert('setting Tab for : ' + tabId);
	var workflowDiv = $('.workflow', '#modal_plan_wizard');
	workflowDiv.find(tabId).show().siblings('div').hide();
	$('.modal-footer', '#modal_plan_wizard').hide();
	workflowDiv.find('li').removeClass('active-tab prev-tab next-tab');
	
	var activeTab = workflowDiv.find('a[href=' + tabId + ']');
	if (!activeTab.exists()) {
		//browser might be returning absolute URI instead of value within href page
		var base = window.location.href.substring(0, window.location.href.lastIndexOf("/") + 1);
		var href = base + tabId;
		// alert('trying to find activeTab using absolute URI: ' + href );
		activeTab = workflowDiv.find("a[href='" + href + "']");
	}
    activeTab.parent().addClass('active-tab').prev().addClass('prev-tab').end().next().addClass('next-tab');
	$(tabId +'-footer', '#modal_plan_wizard').show();
}