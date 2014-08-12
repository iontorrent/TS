support_form_processing = false;
$(function(){

	var show_upload = function (data) {
		$("#support_contact_email").text(data.contact_email);
		$("#support_description").text(data.description);
		$(".support_status").text(data.local_status);
        var created = new Date(data.created);
		$("#support_created").text(created.toLocaleString());
        if (data.ticket_id) {
            $("#ticket_info").show();
            $("#ticket_id").text(data.ticket_id);
            $("#ticket_status").text(data.ticket_status);
            if (data.ticket_message) {
                $("#ticket_message").show().text(data.ticket_message);
            }
        }
	}

    var is_status_done = function (data) {
        var stat = data.local_status;
        return stat == "Complete" || stat == "Access Denied";
    }

	var query_upload = function(url) {
		$.get(url, function(data) {
			show_upload(data);
			if (is_status_done(data))  clearInterval(refresh_timer);
		});
	}

	check_for_upload = function (pk) {
		var url = '/rundb/api/v1/supportupload/?result=' + pk + '&order_by=-id&limit=1';
		$.get(url, function(data){
			if (data.objects.length >= 1) {
				var upload = data.objects[0];
				show_upload(upload);
				if (!is_status_done(upload))  poll_upload(upload);
				$("#support_upload_track").show();
				$("#support_upload_start").hide();
			}
		});
	}

	var handle_errors = function(data) {
		if(data.error == "invalid_auth") {
			$("#support_errors").html('<div class="alert alert-error">Your Torrent Server was not able to authenticate with the Ion Torrent support.</div>');
		} else if(data.error == "invalid_form") {
			for(var input in data.form_errors) {
				$("#"+input).find(".errors")
					.addClass("alert")
					.html(data.form_errors[input]);
			}
		}
	}

	var poll_upload = function(data) {
		refresh_timer = setInterval(function(){query_upload(data.resource_uri)}, 300);
	}

	var setup_submit = function () {
		support_form_processing = true;
		$("#support_errors").html('');
		$("#support_form").find(".errors").removeClass("alert").html('');
		$("#support_form_submit").attr("disabled", "disabled").find("span").text("Submitting...");
	};

	var clean_after_submit = function () {
		support_form_processing = false;
		$("#support_form_submit").removeAttr("disabled").find("span").text("Upload");
	};

	$("#support_form").submit(function(e){
		e.preventDefault();
		if(support_form_processing)
			return;
		setup_submit();
		var form_data = $(this).serialize();
		$.post("/data/export/support/", form_data, function(data){
			if (data.error)
				handle_errors(data);
			else {
				$("#support_upload_track").show();
				$("#support_upload_start").slideUp();
				poll_upload(data);
			}
		}, 'json').fail(function(){
			$("#support_errors").html('<div class="alert alert-error">There was a connection problem submitting your form to the Torrent Server.  Refresh the page and try again.</div>');
		}).always(function(){
			clean_after_submit();
		});
	});
});