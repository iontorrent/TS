# Copyright (C) 2016 Ion Torrent Systems, Inc. All Rights Reserved
# Custom Tastypie APIs to support the TS Mesh
# Wiki page @ https://confluence.amer.thermo.com/x/SwnCBQ

import requests
import multiprocessing
import datetime
import logging

from tastypie.resources import ModelResource
from tastypie.exceptions import InvalidSortError, BadRequest
from django.utils.dateparse import parse_datetime
from django.conf import settings

from ion.utils.TSversion import findVersions
from iondb.rundb.models import IonMeshNode
from iondb.rundb.api import CompositeExperimentResource, IonAuthentication, DjangoAuthorization


def fetch_remote_version_process(new_options):
    """ Used in a multiprocess pool to fetch a TS version for a specific host """
    options = {
        "address": "localhost",
        "resource_name": "torrentsuite",
        "params": {},
    }
    options.update(new_options)

    object = {}
    exceptions = []

    try:
        response = requests.get(
            "http://%s/rundb/api/v1/%s/" % (options["address"], options["resource_name"]),
            params=options["params"])
        response.raise_for_status()
        object = response.json()

    except Exception as e:
        logging.exception(
            "Mesh api failed to fetch version from %s api on %s" % (options["resource_name"], options["address"])
        )
        exceptions.append(e)
    return options["address"], object, exceptions


def fetch_remote_resource_list_process(new_options):
    """ Used in a multiprocess pool to fetch a TS api resource for a specific host """
    options = {
        "address": "localhost",
        "resource_name": "compositeexperiment",
        "params": {},
        "object_limit": 500,
        "page_size": 100,
        "order_by": "-date"
    }
    options.update(new_options)

    options["params"]["limit"] = options["page_size"]
    options["params"]["order_by"] = options["order_by"]

    objects = []
    exceptions = []
    fetched_all_objects = True

    next_url = "http://%s/rundb/api/v1/%s/" % (options["address"], options["resource_name"])
    while next_url:
        try:
            response = requests.get(
                next_url,
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
        except Exception as e:
            logging.exception(
                "Mesh api failed to fetch data from %s api on %s" % (options["resource_name"], options["address"])
            )
            exceptions.append(e)
            fetched_all_objects = False
            next_url = None

    return options["address"], objects, fetched_all_objects, exceptions


class MeshPrefetchResource(ModelResource):
    """ The resource is fetched by the data page before enabled the mesh server drop down. It needs to check versions
    and fetch a list of values for the data page dropdowns. """

    class Meta:
        authentication = IonAuthentication()
        authorization = DjangoAuthorization()

    @staticmethod
    def _fetch_versions(servers):
        _, local_meta_version = findVersions()
        job_arguments = []
        for mesh_node in servers:
            params = {
                "api_key": mesh_node.apikey_remote,
                "system_id": settings.SYSTEM_UUID
            }
            job_arguments.append({
                "address": mesh_node.hostname,
                "params": params
            })
        job_pool = multiprocessing.Pool(processes=len(job_arguments))
        job_output = job_pool.map(fetch_remote_version_process, job_arguments)
        objects_per_host = {}
        for address, object, exceptions in job_output:
            objects_per_host[address] = {
                "object": {},
                "warnings": []
            }
            if exceptions:
                if hasattr(exceptions[0], "response"):
                    response = exceptions[0].response
                    if response.status_code == 401:
                        objects_per_host[address]["warnings"].append("Invalid Permissions")
                else:
                    # Instead of showing this warning, just let the other (incompatible) warning show up on its own.
                    pass

            else:
                objects_per_host[address]["object"] = object.get("meta_version", "")

            if local_meta_version != object.get("meta_version", ""):
                objects_per_host[address]["warnings"].append("Incompatible Software Version")
        return objects_per_host

    @staticmethod
    def _fetch_resource(servers, resource_name, filter_params={}):
        job_arguments = []
        for mesh_node in servers:
            params = {
                "api_key": mesh_node.apikey_remote,
                "system_id": settings.SYSTEM_UUID
            }
            params.update(filter_params)
            job_arguments.append({
                "address": mesh_node.hostname,
                "resource_name": resource_name,
                "object_limit": 1000,
                "order_by": "name",
                "params": params
            })
        job_pool = multiprocessing.Pool(processes=len(job_arguments))
        job_output = job_pool.map(fetch_remote_resource_list_process, job_arguments)
        objects_per_host = {}
        for address, objects, fetched_all_objects, exceptions in job_output:
            objects_per_host[address] = {
                "objects": [],
                "warnings": []
            }
            if exceptions:
                objects_per_host[address]["warnings"].append("Prefetch Failure")
            else:
                objects_per_host[address]["objects"].extend(objects)
        return objects_per_host

    def get_list(self, request, **kwargs):
        container_object = {
            "meta": {},
            "nodes": {},
            "values": {
                "projects": [],
                "samples": [],
                "references": [],
                "rigs": []
            }
        }

        compatible_nodes = list(IonMeshNode.objects.all())

        for node in compatible_nodes:
            container_object["nodes"][node.hostname] = {
                "id": node.id,
                "compatible": True,
                "warnings": []
            }

        # Fetch all mesh versions
        if len(compatible_nodes) > 0:
            versions = self._fetch_versions(
                compatible_nodes
            )
            for host, values in versions.iteritems():
                container_object["nodes"][host]["version"] = values["object"]
                if len(values["warnings"]) > 0:
                    container_object["nodes"][host]["warnings"].extend(values["warnings"])
                    compatible_nodes = [node for node in compatible_nodes if node.hostname != host]
                    container_object["nodes"][host]["compatible"] = False

        # Fetch all mesh projects
        if len(compatible_nodes) > 0:
            projects = self._fetch_resource(
                compatible_nodes,
                "project"
            )
            for host, values in projects.iteritems():
                if len(values["warnings"]) > 0:
                    container_object["nodes"][host]["warnings"].extend(values["warnings"])
                    compatible_nodes = [node for node in compatible_nodes if node.hostname != host]
                    container_object["nodes"][host]["compatible"] = False
                else:
                    for object in values["objects"]:
                        container_object["values"]["projects"].append(object["name"])

        # Fetch all mesh samples
        if len(compatible_nodes) > 0:
            samples = self._fetch_resource(
                compatible_nodes,
                "sample",
                filter_params={"status": "run"}
            )
            for host, values in samples.iteritems():
                if len(values["warnings"]) > 0:
                    container_object["nodes"][host]["warnings"].extend(values["warnings"])
                    compatible_nodes = [node for node in compatible_nodes if node.hostname != host]
                    container_object["nodes"][host]["compatible"] = False
                else:
                    for object in values["objects"]:
                        container_object["values"]["samples"].append(object["name"])

        # Fetch all mesh reference genomes
        if len(compatible_nodes) > 0:
            references = self._fetch_resource(
                compatible_nodes,
                "referencegenome",
                filter_params={"enabled": "true"}
            )
            for host, values in references.iteritems():
                if len(values["warnings"]) > 0:
                    container_object["nodes"][host]["warnings"].extend(values["warnings"])
                    compatible_nodes = [node for node in compatible_nodes if node.hostname != host]
                    container_object["nodes"][host]["compatible"] = False
                else:
                    for object in values["objects"]:
                        container_object["values"]["references"].append(object["short_name"])

        # Fetch all mesh rigs
        if len(compatible_nodes) > 0:
            references = self._fetch_resource(
                compatible_nodes,
                "rig",
            )
            for host, values in references.iteritems():
                if len(values["warnings"]) > 0:
                    container_object["nodes"][host]["warnings"].extend(values["warnings"])
                    compatible_nodes = [node for node in compatible_nodes if node.hostname != host]
                    container_object["nodes"][host]["compatible"] = False
                else:
                    for object in values["objects"]:
                        container_object["values"]["rigs"].append(object["name"])

        # Remove duplicates
        for key in container_object["values"]:
            container_object["values"][key] = list(set(container_object["values"][key]))

        return self.create_response(request, container_object)

    def detail_uri_kwargs(self, bundle_or_obj):
        raise NotImplementedError("This resource only supports listing objects!")

    def obj_get_list(self, bundle, **kwargs):
        return self.get_object_list(bundle.request)

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
        get_args = bundle.request.GET

        applicable_filters = bundle.request.GET.copy()
        applicable_filters.pop("limit", None)
        applicable_filters.pop("offset", None)
        applicable_filters.pop("mesh_node_ids", None)
        applicable_filters.pop("order_by", None)

        if "mesh_node_ids" in get_args:
            mesh_node_ids = get_args["mesh_node_ids"].split(",")
            if "local" in mesh_node_ids:
                include_local_runs = True
                mesh_node_ids.remove("local")
            else:
                include_local_runs = False
            mesh_nodes = IonMeshNode.objects.filter(id__in=mesh_node_ids)
        else:
            include_local_runs = True
            mesh_nodes = IonMeshNode.objects.all()

        job_arguments = []

        if include_local_runs:
            job_arguments.append({
                "address": "localhost",
                "resource_name": "compositeexperiment",
                "object_limit": self._meta.object_limit,
                "params": applicable_filters.copy()
            })
        for mesh_node in mesh_nodes:
            params = applicable_filters.copy()
            params["api_key"] = mesh_node.apikey_remote
            params["system_id"] = settings.SYSTEM_UUID
            job_arguments.append({
                "address": mesh_node.hostname,
                "resource_name": "compositeexperiment",
                "object_limit": self._meta.object_limit,
                "params": params
            })
        job_pool = multiprocessing.Pool(processes=len(job_arguments))
        job_output = job_pool.map(fetch_remote_resource_list_process, job_arguments)

        # Now that we have lists from all the servers we need to check if any had to many objects.
        # If they did, truncate all the lists to the same date range and display it as a warning.

        object_truncation_date = datetime.datetime(2000, 1, 1).date()
        servers_with_truncated_data = []
        servers_with_exceptions = []
        merged_obj_list = []
        warnings = []

        for address, objects, fetched_all_objects, exceptions in job_output:
            if exceptions:
                servers_with_exceptions.append(address)

        # Fist loop, see if any servers have truncated results.
        # If they did, record the truncation date.

        for address, objects, fetched_all_objects, exceptions in job_output:
            if not exceptions:
                if not fetched_all_objects:
                    servers_with_truncated_data.append(address)
                    object_date = parse_datetime(objects[-1]["date"]).date()  # Strip Time
                    if object_date > object_truncation_date:
                        object_truncation_date = object_date

        for address, objects, fetched_all_objects, exceptions in job_output:
            if not exceptions:
                for obj in objects:
                    # Add the _host field
                    if address != "localhost":
                        obj[self._meta.host_field] = address
                    # Add if we are not truncating
                    if len(servers_with_truncated_data) > 0:
                        if parse_datetime(obj["date"]).date() > object_truncation_date:
                            merged_obj_list.append(obj)
                    else:
                        merged_obj_list.append(obj)

        if len(servers_with_truncated_data) > 0:
            warnings.append(
                "The Torrent Server(s) %s have too many results to display. Only experiments newer than %s are displayed!" %
                (",".join(servers_with_truncated_data), str(object_truncation_date))
            )

        if len(servers_with_exceptions) > 0:
            warnings.append(
                "Could not fetch runs from Torrent Server(s) %s!" % ",".join(servers_with_exceptions)
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
