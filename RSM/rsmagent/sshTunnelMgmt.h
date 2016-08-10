// Copyright (C) 2015 Thermo Fisher Scientific. All Rights Reserved.
#ifndef SSH_TUNNEL_MGMT_H
#define SSH_TUNNEL_MGMT_H

int ssh_tunnel_create(
		char const * const host,
		char const * const user,
		char const * const pass,
		const int connectToPort,
		const int tunnelEndpointPort);
void ssh_tunnel_remove();

#endif	// SSH_TUNNEL_MGMT_H
