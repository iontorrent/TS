$(function() {//DOM ready handler
    //----------------------------------------------
	// Clear server name error field
    //----------------------------------------------
	$("#txtServer").keyup(function () {
		$("#hostname_error").html("");
	});

    //----------------------------------------------
	//Disable buttons - until privilege is determined
    //----------------------------------------------
	$("#insufficient_priv").html("<font color=red>You do not have permission to add or remove storage devices</font>");
	$("#btn_remove_mnt").prop('disabled', true);
	$("#btn_mnt_share").prop("disabled", true);
    $("#btn_refresh_devices").prop("disabled", true);
    $("#btn_refresh_mnts").prop("disabled", true);
    $('#txtServer').prop("disabled", true);
    $("#nfs_servers").prop("disabled", true);
    $("#shares_available").prop("disabled", true);
    $("#txt_localmntpoint").prop("disabled", true);
    $("#shares_mounted").prop("disabled", true);
	$.ajax({
		url: 'check_nas_perms/',
		dataType: "json",
		success: function(data){
        	if (data.authorized) {
			    console.log('User has permission to execute');
			    $("#insufficient_priv").html("");
    		    $("#btn_refresh_devices").prop("disabled", false);
    		    $("#btn_refresh_mnts").prop("disabled", false);
    		    $('#txtServer').prop("disabled", false);
    		    $("#nfs_servers").prop("disabled", false);
    		    $("#shares_available").prop("disabled", false);
    		    $("#shares_mounted").prop("disabled", false);
            } else {
			    console.log('User does not have permission to execute');
            }
		},
		error: function(){
				alert("Ajax call failed: check_nas_perms");
		}
    }).done(function(){
        $('#nas_content').unblock();
	});

    //----------------------------------------------
    //Handles selection from server select box
    //----------------------------------------------
    $("#nfs_servers").click(function(){
        var selectedItem = $("#nfs_servers option:selected");
		if (selectedItem.text()) {
			$('#nas_content').block();
			$("#txtServer").val(selectedItem.text());
			ajax_get_avail_mnts(selectedItem.text());
		}
    });

    //-----------------------------------------------
    //Handles Enter key press in txtServer input field
    //-----------------------------------------------
    $('#txtServer').bind("enterKey",function(e){
        var selectedItem = $("#txtServer").val();
		if (selectedItem) {
			$('#nas_content').block();
	        ajax_get_avail_mnts(selectedItem);
		}
    });

    $('#txtServer').keyup(function(e){
        if(e.keyCode == 13)
        {
            $(this).trigger("enterKey");
        }
    });

    //-----------------------------------------------
    //Handles selection from share select box
    //-----------------------------------------------
    $("#shares_available").click(function(){
    	var selectedItem = $("#shares_available option:selected");
		if (selectedItem.text()) {
			sharename = selectedItem.text().split(" ")[0];	//Selects first element of two; the directory share name
			shortId = sharename.replace(/[\\/]/g, "-");		//Replace delimiters with dash to create mountpoint dir name
			var local_mntpoint = "/mnt/nas" + shortId;
			$("#txt_localmntpoint").val(local_mntpoint);
            $("#txt_localmntpoint").prop('disabled', false);
			$("#txt_localmntpoint").keyup();	//Pretend there was a click

			var remote_mntpoint = $("#txtServer").val();
			remote_mntpoint = remote_mntpoint + ":" + sharename;
			$("#txt_remotemntpoint").val(remote_mntpoint);
		}
    });

	//----------------------------------------------
	//Validation for mountpoint directory input
	//----------------------------------------------
	$("#txt_localmntpoint").keyup(function(){
        $("#mntpt_error").html("");
		// Must be valid filename starting with /mnt/
		if (! $("#txt_localmntpoint").val().match(/^(\/mnt\/[A-Za-z0-9_\-\/\.]+$)/)) {
			$("#mntpt_error").html("<font color=red>Invalid directory name</font>");
            $("#btn_mnt_share").prop("disabled", true);
		} else {
        	$("#btn_mnt_share").prop("disabled", false);
        }
	});

    //-----------------------------------------------
    //Handles button click btn_mount_share
    //-----------------------------------------------
    $("#btn_mnt_share").click(function(){
    	var remote_txt = $("#txt_remotemntpoint").val();
        var local_txt = $("#txt_localmntpoint").val();
		if (remote_txt && local_txt) {
			//Send three variables to URL: servername, sharename, mountpoint
			var data_json = {
				"servername": remote_txt.split(":")[0],
				"sharename": remote_txt.split(":")[1],
				"mountpoint": local_txt,
			}
            $('#nas_content').block();
            ajax_add_nas_storage(data_json);
		} else {
			console.log("Something's not defined");
			console.log("remote:'" + remote_txt + "' local:'" + local_txt + "'");
		}
    });

    //------------------------------------------------
    //Handles refresh button click - NAS Devices
	//------------------------------------------------
	$("#btn_refresh_devices").click(function(){
		$("#nfs_servers").empty();
		$("#txtServer").val("");
		$("#shares_available").empty();
		$("#txt_localmntpoint").prop('disabled', true).val("");
		$('#nas_content').block();
		ajax_update_nas_devices();
	});
	//Populate select on page load
	$("#btn_refresh_devices").click();

    //------------------------------------------------
    //Handles refresh button click - mounted volumes
	//------------------------------------------------
	$("#btn_refresh_mnts").click(function(){
		$("#shares_mounted").empty();
		$("#btn_remove_mnt").prop('disabled', true);
		$('#nas_content').block();
		ajax_update_mounted_shares();
	});
	//Populate select on page load
	$("#btn_refresh_mnts").click();

	//------------------------------------------------
	//Handles selection in current volumes selectbox
	//------------------------------------------------
	$("#shares_mounted").click(function() {
		//enable Remove Volume button if there is a valid selection
		var selectedItem = $("#shares_mounted option:selected");
		if (selectedItem.text()) {
			$("#btn_remove_mnt").prop('disabled', false);
		}
	});

    //------------------------------------------------
    //Handles Remove Volume button click
    //------------------------------------------------
	$("#btn_remove_mnt").click(function(){
        var selectedItem = $("#shares_mounted option:selected");
		var servername = selectedItem.text().split(":")[0];
        var mountpoint = selectedItem.text().split(" ")[2];
        console.log("Removing selected volume: " + mountpoint);
        console.log("Removing server: " + servername);
		$('#nas_content').block();
        ajax_remove_volume(servername, mountpoint);
    });

	//------------------------------------------------
	//Ajax call gets list of NAS devices
	//------------------------------------------------
	function ajax_update_nas_devices(){
		console.log("ajax_update_nas_devices");
		$.ajax({
			url: "get_nas_devices/",
			dataType: "json",
        }).done(function(data){
            $('#nas_content').unblock();
			if (data.error.length > 0){
				$("#nfs_servers_error").html("<font color=red>" + data.error + "</font>");
            }
			$("#nfs_servers").empty();
            for (var i in data.devices) {
                console.log(data.devices[i]);
                var temp=$("<option>"+data.devices[i] + "</option>");
                $("#nfs_servers").append(temp);
            }
        }).fail(function() {
			var myerror = "Ajax call failed: ajax_update_nas_devices";
			$("#nfs_servers_error").html("<font color=red>" + myerror + "</font>");
		});
	}

	//------------------------------------------------
	//Ajax call removes selected volume's mount
    //------------------------------------------------
    function ajax_remove_volume(servername, mountpoint){
    	console.log("ajax_remove_volume: "+mountpoint);
        $.ajax({
			url: "remove_nas_storage/",
			type: "POST",
			data: JSON.stringify({"servername": servername, "mountpoint": mountpoint}),
			dataType: "html",
        }).done(function(data){
            $('#nas_content').unblock();
			$("#btn_refresh_mnts").click();
        }).fail(function() {
			alert("Ajax call failed: ajax_remove_volume");
            $('#nas_content').unblock();
		});
    }

	//------------------------------------------------
	//Ajax call gets list of mounted shares and updates display
	//------------------------------------------------
	function ajax_update_mounted_shares() {
		console.log("Current NFS mounted shares");
		$.ajax({
			url: "get_current_mnts/",
			dataType: "json",
        }).done(function(data){
            $('#nas_content').unblock();
			if (data.error.length > 0){
				$("#shares_mounted_error").html("<font color=red>" + data.error + "</font>");
            }
            for (var i in data.mount_dir) {
                console.log(data.mount_dir[i]);
                var temp=$("<option>"+data.mount_dir[i] + "</option>");
                $("#shares_mounted").append(temp);
            }
        }).fail(function() {
			var myerror = "Ajax call failed: ajax_update_mounted_shares";
            $("#shares_mounted_error").html("<font color=red>" + myerror + "</font>");
			$('#nas_content').unblock();
		});
	}

    //------------------------------------------------
	//Ajax call adds storage volume to local filesystem
    //------------------------------------------------
	function ajax_add_nas_storage(data_json) {
		console.log(data_json);
		$.ajax({
			url: "add_nas_storage/",
			type: "POST",
			data: JSON.stringify(data_json),
			dataType: "html",
        }).done(function(data){
            $('#nas_content').unblock();
			$("#btn_refresh_mnts").click();
        }).fail(function() {
			alert("Ajax call failed: ajax_add_nas_storage");
			//TODO: This is the main bit of execution and demands error reporting to the user.
			$('#nas_content').unblock();
		});
	}

    //------------------------------------------------
    //Ajax call gets list of shares for given server and updates display
    //------------------------------------------------
    function ajax_get_avail_mnts(servername) {

		$("#shares_available").empty();
		$("#txt_localmntpoint").prop('disabled', true).val("");
		$("#txt_remotemntpoint").val("");
		$("#hostname_error").val("");
		if ($.trim(servername).length == 0) {
			$('#nas_content').unblock();
			return;
		}

        console.log("Available shares for " + servername);
        $.ajax({
            url: 'get_avail_mnts/' + servername + '/',
            dataType: 'json',
        }).done(function(data){
            $('#nas_content').unblock();
			if (data.error.length > 0) {
				$("#hostname_error").html("<font color=red>" + data.error + "</font>");
			}
			for (var i in data.mount_dir) {
				console.log(data.mount_dir[i]);
				var temp=$("<option>" + data.mount_dir[i] + "</option>");
				$("#shares_available").append(temp);
			}
        }).fail(function() {
			var myerror = "Ajax call failed: ajax_get_avail_mnts";
			$("#hostname_error").html("<font color=red>" + myerror + "</font>");
			$('#nas_content').unblock();
		});
    }
});
