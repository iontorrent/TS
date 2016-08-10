# Copyright (C) 2016 Ion Torrent Systems, Inc. All Rights Reserved
# Custom Tastypie APIs to support the TS Mesh
# Wiki page @ https://confluence.amer.thermo.com/x/SwnCBQ

import requests
import multiprocessing
import datetime

from tastypie.exceptions import InvalidSortError
from django.utils.dateparse import parse_datetime

from iondb.rundb.models import SharedServer
from iondb.rundb.api import CompositeExperimentResource


def fetch_remote_resource_list_process(new_options):
    options = {
        "address": "localhost",
        "resource_name": "compositeexperiment",
        "params": {},
        "auth": ("ionadmin", "ionadmin"),
        "object_limit": 500,
        "page_size": 100
    }
    options.update(new_options)

    options["params"]["limit"] = options["page_size"]
    options["params"]["order_by"] = "-date"

    objects = []
    fetched_all_objects = True
    next_url = "http://%s/rundb/api/v1/%s/" % (options["address"], options["resource_name"])
    while next_url:
        response = requests.get(
            next_url,
            auth=options["auth"],
            params=options["params"])
        response.raise_for_status()
        response_json = response.json()
        objects.extend(response_json["objects"])
        if not response_json["meta"]["next"]:
            next_url = None
        elif len(objects) < options["object_limit"]:
            next_url = ("http://%s/" % options["address"]) + response_json["meta"]["next"]
        else:
            next_url = None
            fetched_all_objects = False
    return options["address"], objects, fetched_all_objects


class MeshCompositeExperimentResource(CompositeExperimentResource):
    class Meta(CompositeExperimentResource.Meta):
        resource_name = CompositeExperimentResource._meta.resource_name
        object_limit = 100
        host_field = "_host"

    def build_schema(self):
        base_schema = super(CompositeExperimentResource, self).build_schema()
        base_schema["fields"][self._meta.host_field] = {
            "blank": False,
            "default": "No default provided.",
            "help_text": "Host this resource is located on.",
            "nullable": False,
            "readonly": True,
            "type": "string",
            "unique": False
        }
        return base_schema

    def get_list(self, request, **kwargs):
        base_bundle = self.build_bundle(request=request)
        objects, warnings = self.obj_get_list(bundle=base_bundle, **self.remove_api_resource_names(kwargs))
        sorted_objects = self.apply_sorting(objects, options=request.GET)

        paginator = self._meta.paginator_class(request.GET, sorted_objects, resource_uri=self.get_resource_uri(),
                                               limit=self._meta.limit, max_limit=self._meta.max_limit,
                                               collection_name=self._meta.collection_name)
        to_be_serialized = paginator.page()

        # Dehydrate the bundles in preparation for serialization.
        bundles = [
            self.full_dehydrate(self.build_bundle(obj=obj, request=request), for_list=True)
            for obj in to_be_serialized[self._meta.collection_name]
            ]

        to_be_serialized[self._meta.collection_name] = bundles
        to_be_serialized = self.alter_list_data_to_serialize(request, to_be_serialized)

        to_be_serialized["warnings"] = warnings

        return self.create_response(request, to_be_serialized)

    def apply_sorting(self, obj_list, options=None):
        if options is None:
            options = {}

        field_name = options.get('order_by', '-date')

        reverse = False
        if field_name[0] == '-':
            reverse = True
            field_name = field_name[1:]

        if field_name not in self.fields:
            raise InvalidSortError("No matching '%s' field for ordering on." % field_name)

        if field_name not in self._meta.ordering:
            raise InvalidSortError("The '%s' field does not allow ordering." % field_name)

        if self.fields[field_name].attribute is None:
            raise InvalidSortError("The '%s' field has no 'attribute' for ordering with." % field_name)

        return sorted(obj_list, key=lambda k: k[self.fields[field_name].attribute], reverse=reverse)

    def full_dehydrate(self, bundle, for_list=False):
        # The object is a dict not and object, just return it.
        return bundle.obj

    def obj_get_list(self, bundle, **kwargs):
        get_args = bundle.request.GET.copy()

        applicable_filters = self.build_filters(filters=get_args)
        if "all_text" in get_args:
            applicable_filters["all_text"] = get_args["all_text"]

        torrent_servers = SharedServer.objects.filter(active=True)

        job_arguments = [{
            "address": "localhost",
            "resource_name": "compositeexperiment",
            "object_limit": self._meta.object_limit,
            "params": applicable_filters
        }]
        for torrent_server in torrent_servers.all():
            job_arguments.append({
                "address": torrent_server.address,
                "resource_name": "compositeexperiment",
                "object_limit": self._meta.object_limit,
                "params": applicable_filters
            })
        job_pool = multiprocessing.Pool(processes=len(job_arguments))
        job_output = job_pool.map(fetch_remote_resource_list_process, job_arguments)

        # Now that we have lists from all the servers we need to check if any had to many objects.
        # If they did, truncate all the lists to the same date range and display it as a warning.

        object_truncation_date = datetime.datetime(2000, 1, 1).date()
        truncated_servers = []

        # Fist loop, see if any servers have truncated results.
        # If they did, record the truncation date.

        for address, objects, fetched_all_objects in job_output:
            if not fetched_all_objects:
                truncated_servers.append(address)
                object_date = parse_datetime(objects[-1]["date"]).date()  # Strip Time
                if object_date > object_truncation_date:
                    object_truncation_date = object_date

        merged_obj_list = []
        warnings = []

        for address, objects, fetched_all_objects in job_output:
            # Add the _host field
            for obj in objects:
                if len(truncated_servers) > 0:
                    if parse_datetime(obj["date"]).date() > object_truncation_date:
                        merged_obj_list.append(obj)
                else:
                    merged_obj_list.append(obj)
                obj[self._meta.host_field] = address
                merged_obj_list.append(obj)

        if len(truncated_servers) > 0:
            warnings.append(
                "The Torrent Server(s) %s have too many results to display. Only experiments newer than %s are displayed!" %
                (",".join(truncated_servers), str(object_truncation_date))
            )

        return merged_obj_list, warnings

    def obj_get(self, bundle, **kwargs):
        raise NotImplementedError("This resource only supports listing objects!")

    def obj_create(self, bundle, **kwargs):
        raise NotImplementedError("This is a readonly resource!")

    def obj_update(self, bundle, **kwargs):
        raise NotImplementedError("This is a readonly resource!")

    def obj_delete_list(self, bundle, **kwargs):
        raise NotImplementedError("This is a readonly resource!")

    def obj_delete(self, bundle, **kwargs):
        raise NotImplementedError("This is a readonly resource!")

    def rollback(self, bundles):
        raise NotImplementedError("This is a readonly resource!")
