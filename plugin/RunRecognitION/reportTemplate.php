<div>
    <div id='message' style="color: red;">
        @success_or_failure_message
    </div>

    <script type='text/javascript'>
    <!--
        var numExecutions = 0;
        var oldHeight = 0;
        function calcHeight() {
            theFrame = document.getElementById('leaderboard_frame')
            theHeight = theFrame.contentWindow.document.body.scrollHeight;
            if (theHeight > oldHeight) {
                theFrame.height = theHeight + 20 + 'px';
                oldHeight = theHeight + 20;
            }
            
            numExecutions++;
            if (numExecutions < 20) {
                setTimeout("calcHeight()", 100);
            }
        }
    //-->
    </script>
<?php
$path = dirname($_SERVER['SCRIPT_NAME']);
echo "<iframe id='leaderboard_frame' src='$path/leaderboard.html' width='100%' frameborder='0' onLoad='calcHeight();'></iframe>";
?>
</div>