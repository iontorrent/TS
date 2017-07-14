# Copyright (C) 2016 Ion Torrent Systems, Inc. All Rights Reserved
import logging
import json
import httplib2
import urlparse
from django.conf import settings
from iondb.rundb.plan.ampliseq import AmpliSeqPanelImport
from iondb.utils.utils import convert

logger = logging.getLogger(__name__)

"""
 This parses all the existing designs, both user created(logged in User) panesls
 and fixed designs panels available in AmpliSeq.com
 if pipeline(DNA/RNA/Exome) is chosen, the related panels will be displayed to import
"""

def match(design, pipeline):
    pipe_types = {
        "RNA": "AMPS_RNA",
        "DNA": "AMPS",
        "exome": "AMPS_EXOME"
    }
    target = pipe_types.get(pipeline, None)
    return not target or design['plan']['runType'] == target

def get_ampliseq_designs(user, password, pipeline, ctx):
    http = httplib2.Http(disable_ssl_certificate_validation=settings.DEBUG)
    http.add_credentials(user, password)
    url = urlparse.urljoin(settings.AMPLISEQ_URL, "ws/design/list")
    response, content = http.request(url)
    if response['status'] == '200':
        design_data = json.loads(content)
        designs = design_data.get('AssayDesigns', [])
        ctx['unordered_solutions'] = []
        ctx['ordered_solutions'] = []
        unordered_tmpList = []
        ordered_tmpList = []
        for design in designs:
            for template in design.get('DesignSolutions', []):
                ampliSeq = AmpliSeqPanelImport(data=template)
                solution, meta = ampliSeq.handle_versioned_plans()
                solution_id = solution['id']
                configurationChoices = solution['configuration_choices']
                if match(solution, pipeline):
                    if solution.get('ordered', False):
                        ctx['ordered_solutions'].append((design, solution))
                        ordered_tmpList.append({'configuration_choices': configurationChoices, 'id': solution_id})
                    else:
                        ctx['unordered_solutions'].append((design, solution))
                        unordered_tmpList.append({'configuration_choices': configurationChoices, 'id': solution_id})
            unordered_tmpList = convert(unordered_tmpList)
            ordered_tmpList = convert(ordered_tmpList)
            ctx['unordered_solution'] = json.dumps(unordered_tmpList)
            ctx['ordered_solution'] = json.dumps(ordered_tmpList)
        return response, ctx
    else:
        return response, {}

def get_fixed_designs_list(fixed_design_data, pipeline):
    # creates fixed_solutions: a list of dictionaries with :
    #    Design id : Ready-to-Use ampliseq panel ID
    #    configuration_choices : the instrument/chip types supported by the corresponding panel.
    #    Type : Community or Fixed Panel - used to to link to Ampliseq website
    #    Genome : Supported genome by the specific panel.
    # final_fixed_soln_data : #Lists all the design panels available for the specific pipeline
    # fixed_ids_choices : This is used to compare the parametized files available and show the warnings

    fixed = []
    fixedDesigns = fixed_design_data.get('TemplateDesigns', [])
    for template in fixedDesigns:
        ampliSeq = AmpliSeqPanelImport(data=template)
        data, meta = ampliSeq.handle_versioned_plans()
        fixed.append(data)
    if fixed:
        ordered_solutions = []
        tmpList = []
        tmpFixedsolLists = []
        fixed_solutions = filter(lambda x: x['status'] == "ORDERABLE" and match(x, pipeline), fixed)
        for design in fixed_solutions:
            designID = design['id']
            configurationChoices = design['configuration_choices']
            tmpDict = {'id': designID,
                       'configuration_choices': configurationChoices}

            tmpFixedSolDict = {'id': designID,
                               'genome': design['genome'],
                               "name" : design["name"],
                               "type" : design["type"],
                               }
            tmpList.append(tmpDict)
            tmpFixedsolLists.append(tmpFixedSolDict)
        #This is used to compare the parametized files available and show the warnings
        fixed_ids_choices = json.dumps(convert(tmpList))

        #Lists all the design panels available for the specific pipeline
        final_fixed_soln_data = tmpFixedsolLists

    return final_fixed_soln_data, ordered_solutions, fixed_ids_choices

def get_ampliseq_fixed_designs(user, password, pipeline, ctx):
    try:
        http = httplib2.Http(disable_ssl_certificate_validation=settings.DEBUG)
        http.add_credentials(user, password)
        url = urlparse.urljoin(settings.AMPLISEQ_URL, "ws/tmpldesign/list/active")
        response, content = http.request(url)
        if response['status'] == '200':
            fixed_design_data = json.loads(content)
            fixed_solutions, ordered_solutions, fixed_ids_choices = get_fixed_designs_list(fixed_design_data, pipeline)
            ctx['fixed_ids_choices'] = fixed_ids_choices
            ctx['fixed_solutions'] = fixed_solutions
        else:
            ctx['http_error'] = "Problem in geting asmpliseq fixed solutions"
            logger.debug("Problem ins geting asmpliseq fixed solutions")
    except:
        ctx['http_error'] = "Could not connect to AmpliSeq.com."
        logger.error("There was a unknown error when contacting ampliseq: %s" % response)

    return ctx

