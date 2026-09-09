# CyberSecurity-Power-System

A comprehensive Python-based framework for cybersecurity analysis and protection of electrical power systems, featuring the GridGuard research framework for power grid security monitoring and vulnerability assessment.

## Overview

This repository contains tools, scripts, and research resources for analyzing and securing power systems against cyber threats. It addresses the critical intersection of cybersecurity and power grid infrastructure, with a focus on practical security monitoring and threat detection.

## Features

- **GridGuard Research Framework**: Advanced power system security analysis platform
- **Telemetry Data Processing**: Handle real-time and historical power system data
- **Graph-based Analysis**: IEEE 123-node test feeder models with adjacency matrices
- **Vulnerability Assessment**: Security vulnerability identification and modeling
- **Threat Detection**: Anomaly detection and cyber-attack pattern recognition
- **Time-series Analysis**: Comprehensive telemetry dataset processing

## Project Structure

```
CyberSecurity-Power-System/
├── README.md                    # Project documentation
├── data/                        # Data directory for power system datasets
│   ├── external/ieee123/        # IEEE 123-node test feeder definitions
│   ├── processed/               # Processed telemetry and artifacts
│   │   ├── telemetry_dataset.parquet
│   │   ├── adjacency_matrix.npy
│   │   └── node_metadata.json
│   └── README.md               # Data directory documentation
├── scripts/                     # Processing and utility scripts
│   ├── fetch_ieee123.py        # IEEE test feeder downloader
│   └── generate_dataset.py     # Dataset generation script
├── configs/                     # Configuration files
│   ├── full.yaml               # Full dataset generation config
│   └── quick.yaml              # Quick test config
└── requirements.txt            # Python dependencies
```

## Requirements

- Python 3.x
- Dependencies listed in `requirements.txt`

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Junaid-EEE11/CyberSecurity-Power-System.git
   cd CyberSecurity-Power-System
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Generating Datasets

To regenerate processed datasets and artifacts:

**Full dataset generation:**
```bash
python scripts/generate_dataset.py --config configs/full.yaml
```

**Quick test generation:**
```bash
python scripts/generate_dataset.py --config configs/quick.yaml
```

### Fetching IEEE Test Feeders

```bash
python scripts/fetch_ieee123.py
```

This script downloads the official IEEE 123-node OpenDSS test feeder definition files with SHA-256 integrity verification.

## Data Format

The processed datasets include:

- **telemetry_dataset.parquet**: Time-series power system telemetry data
- **adjacency_matrix.npy**: Graph adjacency matrix for network topology
- **node_metadata.json**: Node configuration and metadata

## Technologies

- **Language**: Python
- **License**: MIT
- **Data Format**: Parquet, NumPy arrays, JSON
- **Primary Domain**: Power Systems, Cybersecurity, Critical Infrastructure

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. When contributing:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

**Junaid-EEE11**

## Support & Issues

For issues, questions, or suggestions, please use the [Issues](https://github.com/Junaid-EEE11/CyberSecurity-Power-System/issues) section of this repository.

## References

- IEEE 123-node Test Feeder: Standard distribution network model for testing
- OpenDSS: Open Distribution System Simulator
- GridGuard Framework: Research-grade power system security analysis platform

---

**Repository Status**: Active Development  
**Last Updated**: 2026-09-09  
**Visibility**: Private Repository
