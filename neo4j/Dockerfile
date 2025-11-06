FROM neo4j:5.15-community

# Install APOC plugin
RUN apt-get update && \
    apt-get install -y wget && \
    wget -P /var/lib/neo4j/plugins https://github.com/neo4j/apoc/releases/download/5.15.0/apoc-5.15.0-core.jar && \
    rm -rf /var/lib/apt/lists/*

# Set default environment variables
ENV NEO4J_AUTH=neo4j/password \
    NEO4J_apoc_export_file_enabled=true \
    NEO4J_apoc_import_file_enabled=true \
    NEO4J_apoc_import_file_use__neo4j__config=true \
    NEO4J_dbms_security_procedures_unrestricted=apoc.*

EXPOSE 7474 7687

