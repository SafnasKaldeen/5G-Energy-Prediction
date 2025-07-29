import React, { useState, useMemo, useRef, useEffect } from "react";
import {
  MapPin,
  Layers,
  Activity,
  Settings,
  Menu,
  X,
  Loader2,
} from "lucide-react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Slider } from "@/components/ui/slider";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";

interface DataPoint {
  id: string;
  name: string;
  latitude: number;
  longitude: number;
  type?: string;
  area?: string;
  utilization_rate?: number;
  ping_speed?: number;
  status?: string;
  battery_count?: number;
  daily_swaps?: number;
  revenue?: number;
}

interface HeatmapProps {
  data?: DataPoint[];
  config?: {
    heatmapField?: string;
    hexagonSize?: number;
    opacity?: number;
    colorScheme?: "heat" | "cool" | "rainbow" | "viridis" | "density";
    showPoints?: boolean;
    showGrid?: boolean;
    mapProvider?:
      | "openstreetmap"
      | "cartodb_dark"
      | "cartodb_light"
      | "satellite";
    aggregationMethod?: "sum" | "average" | "count" | "max";
  };
  className?: string;
  onDataPointClick?: (point: DataPoint) => void;
}

// Hexagonal grid utilities
const HexUtils = {
  // Convert lat/lng to axial coordinates (q, r)
  latLngToAxial: (lat: number, lng: number, hexSize: number) => {
    // Scale factor to convert degrees to hex grid units
    const scale = hexSize * 0.001; // Adjust this for tighter/looser packing
    
    // Convert to hex grid coordinates
    const x = lng / scale;
    const y = lat / scale;
    
    // Convert cartesian to axial coordinates
    const q = (Math.sqrt(3) * x - y) / 3;
    const r = (2 * y) / 3;
    
    return { q: Math.round(q), r: Math.round(r) };
  },

  // Convert axial coordinates back to lat/lng
  axialToLatLng: (q: number, r: number, hexSize: number) => {
    const scale = hexSize * 0.001;
    
    // Convert axial to cartesian
    const x = scale * (Math.sqrt(3) * q + Math.sqrt(3) * r / 2);
    const y = scale * (3 * r / 2);
    
    return { lat: y, lng: x };
  },

  // Generate hexagon vertices around a center point
  getHexagonVertices: (centerLat: number, centerLng: number, size: number) => {
    const vertices = [];
    const sizeInDegrees = size * 0.0008; // Convert size to approximate degrees
    
    for (let i = 0; i < 6; i++) {
      const angle = (i * Math.PI) / 3;
      const lat = centerLat + sizeInDegrees * Math.sin(angle);
      const lng = centerLng + sizeInDegrees * Math.cos(angle);
      vertices.push([lat, lng]);
    }
    
    return vertices;
  },

  // Get distance between two axial coordinates
  axialDistance: (q1: number, r1: number, q2: number, r2: number) => {
    return (Math.abs(q1 - q2) + Math.abs(q1 + r1 - q2 - r2) + Math.abs(r1 - r2)) / 2;
  }
};

const GeoHexHeatmap: React.FC<HeatmapProps> = ({
  data = [],
  config = {},
  className = "",
  onDataPointClick,
}) => {
  const mapRef = useRef<HTMLDivElement>(null);
  const mapInstance = useRef<any>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [leaflet, setLeaflet] = useState<any>(null);
  const [uiCollapsed, setUiCollapsed] = useState(false);

  // Configuration with defaults
  const [heatmapField, setHeatmapField] = useState(
    config.heatmapField || "utilization_rate"
  );
  const [hexagonSize, setHexagonSize] = useState([config.hexagonSize || 30]);
  const [opacity, setOpacity] = useState([config.opacity || 0.8]);
  const [colorScheme, setColorScheme] = useState(
    config.colorScheme || "density"
  );
  const [showPoints, setShowPoints] = useState(config.showPoints ?? true);
  const [showGrid, setShowGrid] = useState(config.showGrid ?? false);
  const [mapProvider, setMapProvider] = useState(
    config.mapProvider || "cartodb_dark"
  );
  const [aggregationMethod, setAggregationMethod] = useState(
    config.aggregationMethod || "average"
  );

  const mockData: DataPoint[] = [
    // Fixed key stations (clusters around Colombo, Kandy, Galle)
    {
      id: "LK001",
      name: "Station Colombo Fort",
      latitude: 6.9271,
      longitude: 79.8612,
      type: "Battery Swap",
      area: "Colombo",
      utilization_rate: 88,
      ping_speed: 95,
      status: "active",
      battery_count: 28,
      daily_swaps: 50,
      revenue: 1400,
    },
    {
      id: "LK002",
      name: "Station Slave Island",
      latitude: 6.9278,
      longitude: 79.8675,
      type: "Battery Swap",
      area: "Colombo",
      utilization_rate: 84,
      ping_speed: 90,
      status: "active",
      battery_count: 22,
      daily_swaps: 43,
      revenue: 1300,
    },
    {
      id: "LK003",
      name: "Station Kandy City Center",
      latitude: 7.2906,
      longitude: 80.6337,
      type: "Battery Swap",
      area: "Kandy",
      utilization_rate: 72,
      ping_speed: 85,
      status: "active",
      battery_count: 18,
      daily_swaps: 37,
      revenue: 1000,
    },
    {
      id: "LK004",
      name: "Station Peradeniya",
      latitude: 7.2559,
      longitude: 80.5912,
      type: "Charging Station",
      area: "Kandy",
      utilization_rate: 65,
      ping_speed: 80,
      status: "maintenance",
      battery_count: 15,
      daily_swaps: 32,
      revenue: 850,
    },
    {
      id: "LK005",
      name: "Station Galle Fort",
      latitude: 6.0346,
      longitude: 80.217,
      type: "Battery Swap",
      area: "Galle",
      utilization_rate: 70,
      ping_speed: 75,
      status: "active",
      battery_count: 19,
      daily_swaps: 35,
      revenue: 900,
    },
    {
      id: "LK006",
      name: "Station Unawatuna",
      latitude: 5.9413,
      longitude: 80.2596,
      type: "Charging Station",
      area: "Galle",
      utilization_rate: 68,
      ping_speed: 72,
      status: "active",
      battery_count: 16,
      daily_swaps: 30,
      revenue: 820,
    },
    {
      id: "LK007",
      name: "Station Trincomalee",
      latitude: 8.587,
      longitude: 81.2152,
      type: "Battery Swap",
      area: "Trincomalee",
      utilization_rate: 60,
      ping_speed: 70,
      status: "warning",
      battery_count: 14,
      daily_swaps: 25,
      revenue: 650,
    },
    {
      id: "LK008",
      name: "Station Batticaloa",
      latitude: 7.712,
      longitude: 81.6784,
      type: "Charging Station",
      area: "Batticaloa",
      utilization_rate: 58,
      ping_speed: 68,
      status: "maintenance",
      battery_count: 13,
      daily_swaps: 23,
      revenue: 600,
    },
    {
      id: "LK009",
      name: "Station Jaffna",
      latitude: 9.6615,
      longitude: 80.0255,
      type: "Battery Swap",
      area: "Jaffna",
      utilization_rate: 50,
      ping_speed: 60,
      status: "active",
      battery_count: 14,
      daily_swaps: 22,
      revenue: 600,
    },
    {
      id: "LK010",
      name: "Station Negombo",
      latitude: 7.2087,
      longitude: 79.835,
      type: "Battery Swap",
      area: "Negombo",
      utilization_rate: 70,
      ping_speed: 74,
      status: "active",
      battery_count: 18,
      daily_swaps: 32,
      revenue: 780,
    },

    // Random generated stations for density (~20)
    ...Array.from({ length: 20 }, (_, i) => ({
      id: `LKGEN${(i + 11).toString().padStart(3, "0")}`,
      name: `Station ${i + 11}`,
      latitude: 5.9 + Math.random() * (9.8 - 5.9), // Sri Lanka lat range
      longitude: 79.7 + Math.random() * (81.9 - 79.7), // Sri Lanka lon range
      type: ["Battery Swap", "Charging Station"][Math.floor(Math.random() * 2)],
      area: "Sri Lanka",
      utilization_rate: Math.floor(Math.random() * 100),
      ping_speed: Math.floor(50 + Math.random() * 70),
      status: ["active", "warning", "maintenance"][
        Math.floor(Math.random() * 3)
      ],
      battery_count: Math.floor(10 + Math.random() * 20),
      daily_swaps: Math.floor(Math.random() * 50),
      revenue: Math.floor(Math.random() * 1500),
    })),
  ];

  const stations = data.length > 0 ? data : mockData;

  // Color schemes
  const colorSchemes = {
    heat: [
      "rgba(0,0,255,0)",
      "rgba(0,255,255,0.8)",
      "rgba(0,255,0,0.8)",
      "rgba(255,255,0,0.8)",
      "rgba(255,0,0,0.8)",
    ],
    cool: [
      "rgba(0,0,0,0)",
      "rgba(102,126,234,0.6)",
      "rgba(118,75,162,0.7)",
      "rgba(240,147,251,0.8)",
      "rgba(245,87,108,0.9)",
    ],
    rainbow: [
      "rgba(0,0,0,0)",
      "rgba(79,172,254,0.6)",
      "rgba(0,242,254,0.7)",
      "rgba(67,233,123,0.8)",
      "rgba(56,249,215,0.9)",
    ],
    viridis: [
      "rgba(0,0,0,0)",
      "rgba(68,1,84,0.6)",
      "rgba(49,104,142,0.7)",
      "rgba(53,183,121,0.8)",
      "rgba(253,231,37,0.9)",
    ],
    density: [
      "rgba(0,0,0,0)",
      "rgba(25,25,112,0.5)",
      "rgba(0,191,255,0.6)",
      "rgba(50,205,50,0.7)",
      "rgba(255,165,0,0.8)",
      "rgba(255,69,0,0.9)",
    ],
  };

  const colors = colorSchemes[colorScheme];

  // Map providers
  const mapProviders = {
    openstreetmap: {
      name: "OpenStreetMap",
      url: "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
      attribution: "© OpenStreetMap contributors",
    },
    cartodb_dark: {
      name: "Carto Dark",
      url: "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
      attribution: "© OpenStreetMap contributors © CARTO",
    },
    cartodb_light: {
      name: "Carto Light",
      url: "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
      attribution: "© OpenStreetMap contributors © CARTO",
    },
    satellite: {
      name: "Satellite",
      url: "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
      attribution: "© Esri",
    },
  };

  // Load Leaflet
  useEffect(() => {
    const loadLeaflet = async () => {
      try {
        const L = await import("leaflet");
        await import("leaflet/dist/leaflet.css");
        setLeaflet(L);
      } catch (error) {
        console.error("Failed to load Leaflet:", error);
        setIsLoading(false);
      }
    };
    loadLeaflet();
  }, []);

  // Generate tightly packed hexagonal grid and aggregate data
  const hexagonalData = useMemo(() => {
    if (stations.length === 0) return [];

    const hexagons = new Map<string, any>();
    const hexSize = hexagonSize[0];

    // Convert each station to hexagonal grid coordinates and aggregate
    stations.forEach((station) => {
      const { q, r } = HexUtils.latLngToAxial(station.latitude, station.longitude, hexSize);
      const key = `${q},${r}`;

      if (!hexagons.has(key)) {
        const { lat, lng } = HexUtils.axialToLatLng(q, r, hexSize);
        hexagons.set(key, {
          q,
          r,
          centerLat: lat,
          centerLng: lng,
          points: [],
          value: 0,
          count: 0,
        });
      }

      const hex = hexagons.get(key);
      hex.points.push(station);
      hex.count += 1;

      const fieldValue = (station[heatmapField as keyof DataPoint] as number) || 0;
      
      switch (aggregationMethod) {
        case "sum":
          hex.value += fieldValue;
          break;
        case "average":
          hex.value = hex.points.reduce(
            (sum: number, p: DataPoint) =>
              sum + ((p[heatmapField as keyof DataPoint] as number) || 0),
            0
          ) / hex.points.length;
          break;
        case "count":
          hex.value = hex.count;
          break;
        case "max":
          hex.value = Math.max(hex.value, fieldValue);
          break;
        default:
          hex.value += fieldValue;
      }
    });

    return Array.from(hexagons.values());
  }, [stations, heatmapField, hexagonSize, aggregationMethod]);

  // Initialize map
  useEffect(() => {
    if (!leaflet || !mapRef.current) return;

    const L = leaflet.default || leaflet;

    if (!mapInstance.current) {
      mapInstance.current = L.map(mapRef.current, {
        zoomControl: true,
        dragging: true,
        scrollWheelZoom: true,
      });
    }

    // Set view to Sri Lanka
    mapInstance.current.setView([7.8731, 80.7718], 7);

    // Remove existing layers
    mapInstance.current.eachLayer((layer: any) => {
      mapInstance.current.removeLayer(layer);
    });

    // Add tile layer
    const provider = mapProviders[mapProvider];
    const tileOptions: any = {
      attribution: provider.attribution,
      maxZoom: 19,
    };

    if (provider.url.includes("{s}")) {
      tileOptions.subdomains = "abcd";
    }

    L.tileLayer(provider.url, tileOptions).addTo(mapInstance.current);

    // Add tightly packed hexagonal heatmap
    if (hexagonalData.length > 0) {
      const maxValue = Math.max(...hexagonalData.map((h) => h.value));
      const minValue = Math.min(...hexagonalData.map((h) => h.value));

      hexagonalData.forEach((hex) => {
        if (hex.value > 0) {
          const normalizedValue =
            maxValue > minValue
              ? (hex.value - minValue) / (maxValue - minValue)
              : 0.5;
          
          const colorIndex = Math.min(
            Math.floor(normalizedValue * (colors.length - 1)),
            colors.length - 1
          );
          const color = colors[colorIndex] || colors[0];

          // Generate perfect hexagon vertices
          const hexVertices = HexUtils.getHexagonVertices(
            hex.centerLat,
            hex.centerLng,
            hexagonSize[0]
          );

          const hexPolygon = L.polygon(hexVertices, {
            fillColor: color.replace(
              /rgba\(([^)]+)\)/,
              (match: string, p1: string) => {
                const [r, g, b] = p1.split(",").map(Number);
                return `rgb(${r}, ${g}, ${b})`;
              }
            ),
            fillOpacity: normalizedValue * opacity[0],
            stroke: showGrid,
            color: showGrid ? "#ffffff" : "transparent",
            weight: showGrid ? 1 : 0,
            opacity: showGrid ? 0.3 : 0,
          }).addTo(mapInstance.current);

          // Add popup with aggregated data
          const popupContent = `
            <div style="color: white; font-family: system-ui, sans-serif;">
              <div style="font-weight: 600; margin-bottom: 8px;">Hex Grid (${hex.q}, ${hex.r})</div>
              <div style="font-size: 13px;">
                <div>Points: ${hex.count}</div>
                <div>${heatmapField.replace("_", " ")}: ${hex.value.toFixed(1)}</div>
                <div>Method: ${aggregationMethod}</div>
                <div>Intensity: ${(normalizedValue * 100).toFixed(1)}%</div>
              </div>
            </div>
          `;

          hexPolygon.bindPopup(popupContent, {
            className: "custom-popup",
          });
        }
      });
    }

    // Add individual points if enabled
    if (showPoints) {
      stations.forEach((station) => {
        const marker = L.circleMarker([station.latitude, station.longitude], {
          radius: 4,
          fillColor: "#ffffff",
          color: "#1e293b",
          weight: 2,
          opacity: 0.9,
          fillOpacity: 0.8,
        }).addTo(mapInstance.current);

        const popupContent = `
          <div style="color: white; font-family: system-ui, sans-serif;">
            <div style="font-weight: 600; margin-bottom: 8px;">${station.name}</div>
            <div style="font-size: 13px;">
              <div>${heatmapField.replace("_", " ")}: ${
          (station[heatmapField as keyof DataPoint] as number) || "N/A"
        }</div>
              <div>Location: ${station.latitude.toFixed(4)}, ${station.longitude.toFixed(4)}</div>
              ${station.status ? `<div>Status: ${station.status}</div>` : ""}
            </div>
          </div>
        `;

        marker.bindPopup(popupContent, {
          className: "custom-popup",
        });

        marker.on("click", () => {
          if (onDataPointClick) onDataPointClick(station);
        });
      });
    }

    setIsLoading(false);
  }, [
    leaflet,
    hexagonalData,
    showPoints,
    showGrid,
    mapProvider,
    opacity,
    colors,
    hexagonSize,
  ]);

  const stats = useMemo(() => {
    if (hexagonalData.length === 0)
      return { avg: 0, max: 0, min: 0, totalHexagons: 0 };

    const values = hexagonalData.map((h) => h.value).filter((v) => v > 0);
    return {
      avg: values.reduce((sum, v) => sum + v, 0) / values.length,
      max: Math.max(...values),
      min: Math.min(...values),
      totalHexagons: values.length,
    };
  }, [hexagonalData]);

  return (
    <div
      className={`relative w-full h-screen bg-background overflow-hidden ${className}`}
    >
      {/* Header */}
      <div className="absolute top-4 left-4 z-[999]">
        <Card className="bg-card/95 backdrop-blur-sm border border-border/50 p-3">
          <div className="flex items-center space-x-2">
            <Activity className="h-5 w-5 text-primary" />
            <span className="text-foreground font-medium">
              Tightly Packed Hex Grid
            </span>
          </div>
          <div className="text-xs text-muted-foreground mt-1">
            {stats.totalHexagons} hexagons • {stations.length} points
          </div>
        </Card>
      </div>

      {/* Settings Toggle */}
      <div className="absolute top-4 right-4 z-[999]">
        <Button
          variant="outline"
          size="sm"
          onClick={() => setUiCollapsed(!uiCollapsed)}
          className="bg-card/95 backdrop-blur-sm border border-border/50"
        >
          {uiCollapsed ? (
            <Menu className="h-4 w-4" />
          ) : (
            <X className="h-4 w-4" />
          )}
        </Button>
      </div>

      {/* Settings Panel */}
      {!uiCollapsed && (
        <div className="absolute top-16 right-4 z-[998] max-w-xs">
          <Card className="bg-card/95 backdrop-blur-sm border border-border/50 p-4">
            <div className="text-sm font-semibold text-foreground mb-3 flex items-center gap-2">
              <Settings className="h-4 w-4" />
              Heatmap Settings
            </div>
            <div className="space-y-4 text-xs">
              <div className="space-y-2">
                <label className="text-foreground block text-xs font-medium">
                  Data Field
                </label>
                <Select value={heatmapField} onValueChange={setHeatmapField}>
                  <SelectTrigger className="h-8 text-xs">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="utilization_rate">
                      Utilization Rate
                    </SelectItem>
                    <SelectItem value="daily_swaps">Daily Swaps</SelectItem>
                    <SelectItem value="revenue">Revenue</SelectItem>
                    <SelectItem value="battery_count">Battery Count</SelectItem>
                    <SelectItem value="ping_speed">Ping Speed</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <label className="text-foreground block text-xs font-medium">
                  Aggregation: {aggregationMethod}
                </label>
                <Select
                  value={aggregationMethod}
                  onValueChange={(value) =>
                    setAggregationMethod(
                      value as "sum" | "average" | "count" | "max"
                    )
                  }
                >
                  <SelectTrigger className="h-8 text-xs">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="average">Average</SelectItem>
                    <SelectItem value="sum">Sum</SelectItem>
                    <SelectItem value="count">Count</SelectItem>
                    <SelectItem value="max">Maximum</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <label className="text-foreground block text-xs font-medium">
                  Hexagon Size: {hexagonSize[0]}px
                </label>
                <Slider
                  value={hexagonSize}
                  onValueChange={setHexagonSize}
                  max={100}
                  min={10}
                  step={5}
                  className="w-full"
                />
              </div>

              <div className="space-y-2">
                <label className="text-foreground block text-xs font-medium">
                  Opacity: {Math.round(opacity[0] * 100)}%
                </label>
                <Slider
                  value={opacity}
                  onValueChange={setOpacity}
                  max={1}
                  min={0.1}
                  step={0.1}
                  className="w-full"
                />
              </div>

              <div className="space-y-2">
                <label className="text-foreground block text-xs font-medium">
                  Color Scheme
                </label>
                <Select
                  value={colorScheme}
                  onValueChange={(value) =>
                    setColorScheme(
                      value as
                        | "heat"
                        | "cool"
                        | "rainbow"
                        | "viridis"
                        | "density"
                    )
                  }
                >
                  <SelectTrigger className="h-8 text-xs">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="density">Density</SelectItem>
                    <SelectItem value="heat">Heat</SelectItem>
                    <SelectItem value="cool">Cool</SelectItem>
                    <SelectItem value="rainbow">Rainbow</SelectItem>
                    <SelectItem value="viridis">Viridis</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <label className="text-foreground block text-xs font-medium">
                  Map Style
                </label>
                <Select
                  value={mapProvider}
                  onValueChange={(value) =>
                    setMapProvider(
                      value as
                        | "openstreetmap"
                        | "cartodb_dark"
                        | "cartodb_light"
                        | "satellite"
                    )
                  }
                >
                  <SelectTrigger className="h-8 text-xs">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="cartodb_dark">Dark</SelectItem>
                    <SelectItem value="cartodb_light">Light</SelectItem>
                    <SelectItem value="openstreetmap">OpenStreetMap</SelectItem>
                    <SelectItem value="satellite">Satellite</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="flex items-center justify-between">
                <label className="text-foreground text-xs font-medium">
                  Show Points
                </label>
                <Switch checked={showPoints} onCheckedChange={setShowPoints} />
              </div>

              <div className="flex items-center justify-between">
                <label className="text-foreground text-xs font-medium">
                  Show Grid Lines
                </label>
                <Switch checked={showGrid} onCheckedChange={setShowGrid} />
              </div>
            </div>
          </Card>
        </div>
      )}

      {/* Legend */}
      <div className="absolute bottom-4 right-4 z-[999]">
        <Card className="bg-card/95 backdrop-blur-sm border border-border/50 p-3">
          <div className="text-sm font-medium text-foreground mb-2">
            Density Scale
          </div>
          <div className="flex items-center space-x-2">
            <span className="text-xs text-muted-foreground">Low</span>
            <div
              className="w-20 h-3 rounded-full"
              style={{
                background: `linear-gradient(to right, ${colors
                  .slice(1)
                  .join(", ")})`,
              }}
            />
            <span className="text-xs text-muted-foreground">High</span>
          </div>
          <div className="text-xs text-muted-foreground mt-1">
            {heatmapField.replace("_", " ").toUpperCase()}
          </div>
        </Card>
      </div>

      {/* Stats */}
      <div className="absolute bottom-4 left-4 z-[999]">
        <Card className="bg-card/95 backdrop-blur-sm border border-border/50 p-3">
          <div className="text-xs text-muted-foreground space-y-1">
            <div>Avg: {stats.avg.toFixed(1)}</div>
            <div>Max: {stats.max.toFixed(1)}</div>
            <div>Min: {stats.min.toFixed(1)}</div>
          </div>
        </Card>
      </div>

      {/* Map Container */}
      <div className="absolute inset-0">
        {isLoading && (
          <div className="absolute inset-0 flex items-center justify-center bg-background/80 z-[1001]">
            <div className="flex flex-col items-center text-foreground">
              <Loader2 className="h-8 w-8 animate-spin mb-2" />
              <p className="text-sm">Loading heatmap...</p>
            </div>
          </div>
        )}
        <div ref={mapRef} className="h-full w-full" />
      </div>

      {/* No Data State */}
      {stations.length === 0 && !isLoading && (
        <div className="absolute inset-0 flex items-center justify-center bg-background/80 backdrop-blur-sm z-[1002]">
          <div className="text-center text-muted-foreground p-6">
            <MapPin className="h-12 w-12 mx-auto mb-4 opacity-50" />
            <div className="text-lg font-medium mb-2">No Data Available</div>
            <div className="text-sm">
              Add data points to display density heatmap
            </div>
          </div>
        </div>
      )}

      {/* Custom popup styles */}
      <style
        dangerouslySetInnerHTML={{
          __html: `
          .custom-popup .leaflet-popup-content-wrapper {
            background-color: hsl(var(--card));
            border: 1px solid hsl(var(--border));
            border-radius: 12px;
            color: hsl(var(--foreground));
            backdrop-filter: blur(12px);
          }
          .custom-popup .leaflet-popup-content {
            margin: 12px;
            color: hsl(var(--foreground));
          }
          .custom-popup .leaflet-popup-tip {
            background-color: hsl(var(--card));
            border: 1px solid hsl(var(--border));
          }
          .custom-popup a.leaflet-popup-close-button {
            color: hsl(var(--muted-foreground));
            font-size: 18px;
            padding: 4px 8px;
            border-radius: 4px;
          }
          .custom-popup a.leaflet-popup-close-button:hover {
            color: hsl(var(--foreground));
            background-color: hsl(var(--accent));
          }
        `,
        }}
      />
    </div>
  );
};

export default GeoHexHeatmap;