#!/usr/bin/env node
/**
 * UDP/TCP relay: Galaxy XR → Surface (10.0.0.32) → DGX Spark LiveKit WebRTC.
 * Required because XR cannot reach Spark directly (router client isolation).
 * Pair with XR_ICE_ADVERTISE_IP=10.0.0.32 on the Spark LiveKit container.
 */
import dgram from 'node:dgram';
import net from 'node:net';

const SPARK_HOST = process.env.XR_SPARK_HOST || '10.0.0.158';
const UDP_LISTEN = Number(process.env.XR_RELAY_UDP_PORT || 7882);
const UDP_TARGET = Number(process.env.XR_SPARK_UDP_PORT || 7882);
const TCP_LISTEN = Number(process.env.XR_RELAY_TCP_PORT || 7881);
const TCP_TARGET = Number(process.env.XR_SPARK_TCP_PORT || 7881);

/** @type {Map<string, import('node:dgram').Socket>} */
const udpSessions = new Map();
const UDP_IDLE_MS = 60_000;

function udpKey(rinfo) {
  return `${rinfo.address}:${rinfo.port}`;
}

const udpServer = dgram.createSocket('udp4');
udpServer.on('message', (msg, rinfo) => {
  const key = udpKey(rinfo);
  let upstream = udpSessions.get(key);
  if (!upstream) {
    upstream = dgram.createSocket('udp4');
    upstream.on('message', (reply) => {
      udpServer.send(reply, rinfo.port, rinfo.address);
    });
    upstream.on('error', () => {
      upstream?.close();
      udpSessions.delete(key);
    });
    udpSessions.set(key, upstream);
    setTimeout(() => {
      if (!udpSessions.has(key)) return;
      udpSessions.get(key)?.close();
      udpSessions.delete(key);
    }, UDP_IDLE_MS);
  }
  upstream.send(msg, UDP_TARGET, SPARK_HOST);
});
udpServer.on('error', (err) => {
  console.error('UDP relay error:', err.message);
  process.exit(1);
});

const tcpServer = net.createServer((client) => {
  const upstream = net.connect(TCP_TARGET, SPARK_HOST);
  client.pipe(upstream);
  upstream.pipe(client);
  const cleanup = () => {
    client.destroy();
    upstream.destroy();
  };
  client.on('error', cleanup);
  upstream.on('error', cleanup);
});
tcpServer.on('error', (err) => {
  console.error('TCP relay error:', err.message);
  process.exit(1);
});

udpServer.bind(UDP_LISTEN, '0.0.0.0', () => {
  console.log(`XR media relay UDP 0.0.0.0:${UDP_LISTEN} → ${SPARK_HOST}:${UDP_TARGET}`);
});
tcpServer.listen(TCP_LISTEN, '0.0.0.0', () => {
  console.log(`XR media relay TCP 0.0.0.0:${TCP_LISTEN} → ${SPARK_HOST}:${TCP_TARGET}`);
});
