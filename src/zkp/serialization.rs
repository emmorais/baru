// Serialization and storage for ZKP proofs and circuits
// Handles saving/loading proofs and circuits to/from JSON

use super::{Circuit, CircuitMetadata, PublicValues, ZKProof};
use serde_json::{json, Value};
use std::fs;
use std::path::Path;

/// Save a proof to a JSON file
pub fn save_proof<P: AsRef<Path>>(proof: &ZKProof, path: P) -> Result<(), String> {
    let json = json!({
        "backend": proof.backend_name,
        "proof_data": hex::encode(&proof.proof_data),
        "public_values": proof.public_values,
        "timestamp": proof.timestamp,
    });

    let json_string = serde_json::to_string_pretty(&json)
        .map_err(|e| format!("Failed to serialize proof: {}", e))?;

    fs::write(&path, json_string)
        .map_err(|e| format!("Failed to write proof file: {}", e))
}

/// Load a proof from a JSON file
pub fn load_proof<P: AsRef<Path>>(path: P) -> Result<ZKProof, String> {
    let json_string = fs::read_to_string(&path)
        .map_err(|e| format!("Failed to read proof file: {}", e))?;

    let json: Value = serde_json::from_str(&json_string)
        .map_err(|e| format!("Failed to parse proof JSON: {}", e))?;

    let backend_name = json["backend"]
        .as_str()
        .ok_or("Missing 'backend' field in proof")?
        .to_string();

    let proof_data_hex = json["proof_data"]
        .as_str()
        .ok_or("Missing 'proof_data' field in proof")?;

    let proof_data = hex::decode(proof_data_hex)
        .map_err(|e| format!("Failed to decode proof data: {}", e))?;

    let public_values = json["public_values"]
        .as_object()
        .ok_or("Missing 'public_values' field in proof")?
        .iter()
        .map(|(k, v)| {
            let val = v.as_i64().ok_or("Invalid value in public_values")?;
            Ok((k.clone(), val))
        })
        .collect::<Result<PublicValues, String>>()?;

    let timestamp = json["timestamp"]
        .as_u64()
        .unwrap_or(0);

    Ok(ZKProof {
        proof_data,
        public_values,
        backend_name,
        timestamp,
    })
}

/// Save a circuit to a JSON file
pub fn save_circuit<P: AsRef<Path>>(circuit: &Circuit, path: P) -> Result<(), String> {
    let json = json!({
        "circuit_data": hex::encode(&circuit.circuit_data),
        "witness_vars": circuit.witness_vars,
        "public_vars": circuit.public_vars,
        "metadata": {
            "num_gates": circuit.metadata.num_gates,
            "num_constraints": circuit.metadata.num_constraints,
            "backend_name": circuit.metadata.backend_name,
            "timestamp": circuit.metadata.timestamp,
        }
    });

    let json_string = serde_json::to_string_pretty(&json)
        .map_err(|e| format!("Failed to serialize circuit: {}", e))?;

    fs::write(&path, json_string)
        .map_err(|e| format!("Failed to write circuit file: {}", e))
}

/// Load a circuit from a JSON file
pub fn load_circuit<P: AsRef<Path>>(path: P) -> Result<Circuit, String> {
    let json_string = fs::read_to_string(&path)
        .map_err(|e| format!("Failed to read circuit file: {}", e))?;

    let json: Value = serde_json::from_str(&json_string)
        .map_err(|e| format!("Failed to parse circuit JSON: {}", e))?;

    let circuit_data_hex = json["circuit_data"]
        .as_str()
        .ok_or("Missing 'circuit_data' field")?;

    let circuit_data = hex::decode(circuit_data_hex)
        .map_err(|e| format!("Failed to decode circuit data: {}", e))?;

    let witness_vars: Vec<String> = json["witness_vars"]
        .as_array()
        .ok_or("Missing 'witness_vars' field")?
        .iter()
        .map(|v| {
            v.as_str()
                .ok_or("Invalid witness_var".to_string())
                .map(|s| s.to_string())
        })
        .collect::<Result<Vec<_>, _>>()?;

    let public_vars: Vec<String> = json["public_vars"]
        .as_array()
        .ok_or("Missing 'public_vars' field")?
        .iter()
        .map(|v| {
            v.as_str()
                .ok_or("Invalid public_var".to_string())
                .map(|s| s.to_string())
        })
        .collect::<Result<Vec<_>, _>>()?;

    let metadata = CircuitMetadata {
        num_gates: json["metadata"]["num_gates"]
            .as_u64()
            .unwrap_or(0) as usize,
        num_constraints: json["metadata"]["num_constraints"]
            .as_u64()
            .unwrap_or(0) as usize,
        backend_name: json["metadata"]["backend_name"]
            .as_str()
            .ok_or("Missing backend_name in metadata")?
            .to_string(),
        timestamp: json["metadata"]["timestamp"]
            .as_u64()
            .unwrap_or(0),
    };

    Ok(Circuit {
        circuit_data,
        witness_vars,
        public_vars,
        metadata,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use tempfile::NamedTempFile;

    #[test]
    fn test_save_load_proof() {
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path();

        let mut public_values = HashMap::new();
        public_values.insert("x".to_string(), 42);
        public_values.insert("y".to_string(), 100);

        let proof = ZKProof {
            proof_data: vec![1, 2, 3, 4],
            public_values,
            backend_name: "binius64".to_string(),
            timestamp: 12345,
        };

        // Save
        save_proof(&proof, path).unwrap();

        // Load
        let loaded = load_proof(path).unwrap();

        assert_eq!(loaded.backend_name, "binius64");
        assert_eq!(loaded.proof_data, vec![1, 2, 3, 4]);
        assert_eq!(loaded.timestamp, 12345);
        assert_eq!(loaded.public_values.get("x"), Some(&42));
        assert_eq!(loaded.public_values.get("y"), Some(&100));
    }

    #[test]
    fn test_save_load_circuit() {
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path();

        let circuit = Circuit {
            circuit_data: vec![0xDE, 0xAD, 0xBE, 0xEF],
            witness_vars: vec!["secret".to_string(), "key".to_string()],
            public_vars: vec!["result".to_string()],
            metadata: CircuitMetadata {
                num_gates: 100,
                num_constraints: 50,
                backend_name: "binius64".to_string(),
                timestamp: 54321,
            },
        };

        // Save
        save_circuit(&circuit, path).unwrap();

        // Load
        let loaded = load_circuit(path).unwrap();

        assert_eq!(loaded.circuit_data, vec![0xDE, 0xAD, 0xBE, 0xEF]);
        assert_eq!(loaded.witness_vars, vec!["secret", "key"]);
        assert_eq!(loaded.public_vars, vec!["result"]);
        assert_eq!(loaded.metadata.num_gates, 100);
        assert_eq!(loaded.metadata.num_constraints, 50);
        assert_eq!(loaded.metadata.backend_name, "binius64");
    }
}
