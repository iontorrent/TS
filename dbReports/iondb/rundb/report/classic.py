# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
from iondb.rundb import models
from django import shortcuts, template
from django.contrib.auth.decorators import login_required
import os
import json

from iondb.rundb.views import barcodeData


@login_required
def graph_iframe(request, pk):
    """
    Make a Protovis graph from the requested metric,
    !!! Used by Default_Report.php (TS/pipeline/web/db/writers/combinedReport.php, TS/pipeline/web/db/writers/format_whole.php)
    !!! Similar functionality exists in ResultsResources.get_barcode()
    """
    metric = request.GET.get('metric', False)

    result = shortcuts.get_object_or_404(models.Results, pk=pk)

    barcodeSummary = "alignment_barcode_summary.csv"
    data = barcodeData(os.path.join(result.get_report_dir(), barcodeSummary), metric)

    ctxd = {"data": json.dumps(data)}
    context = template.RequestContext(request, ctxd)

    return shortcuts.render_to_response("rundb/reports/classic/ion_graph_iframe.html", context_instance=context)
