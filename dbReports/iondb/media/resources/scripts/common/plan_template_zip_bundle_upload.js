$(function() {
  uploader = new plupload.Uploader({
    container : 'bedPublish',
    max_file_size : '1000mb',
    runtimes: 'html5,flash,silverlight',
    browse_button: 'pickfiles',
    url: '/rundb/publish/plupload/BED/',
    chunk_size: '10mb',
    unique_names: false,
    multi_selection: false,
    multiple_queues: false,
    multipart_params: {meta: '{}'},
    silverlight_xap_url: '{% static "resources/plupload/js/Moxie.xap"%}',
    flash_swf_url: '{% static "resources/plupload/js/Moxie.swf"%}'
  });

  uploader.bind('Init', function(up, params) {
    // Don't do anyting special on init.
  });

  $('#upload_button').click(function() {
    uploader.settings.multipart_params.meta = JSON.stringify({
      hotspot: false,
      reference: null,
      choice: $('input:radio[name=instrument_choice]:checked').val()
    });
    uploader.start();
    return false;
  });

  uploader.init();

  uploader.bind('FilesAdded', function(up, files) {
    var file = files[0];
    if (up.files.length > 1) {
      up.removeFile(up.files[0]);
    }
    console.log(up.files);
    $('#file_info').html(file.name + ' (' + plupload.formatSize(file.size) + ')');
    $('#file_progress').html('');
    up.refresh(); // Reposition Flash/Silverlight
  });

  uploader.bind('UploadProgress', function(up, file) {
    $('#file_progress').html(file.percent + "%");
  });

  uploader.bind('Error', function(up, err) {
    $('#file_info').append("<div>There was an error during your upload.  Please refresh the page and try again.</div>");
  });

  uploader.bind('FileUploaded', function(up, file) {
    $('#file_progress').html("100%").delay(500).html("");
    $("#file_info").delay(500).html("File uploaded successfully.  Upload another or check the appropriate reference page for processing status.")
  });
  $('#hotspot_help_text').hide();
  $('#hotspot_help_button').click(function(){
    $('#hotspot_help_text').slideToggle(300);
    return false;
  });
});