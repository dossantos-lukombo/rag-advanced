export $(grep -v '^#' ../.env | xargs)

docker stop $(docker ps -l -q)

docker run -d \
    --publish=7474:7474 --publish=7687:7687 \
    --volume=$HOME/Documents/projets_perso/Code/rag-advanced/neo4j/data:/data \
    --env NEO4J_AUTH=$USERNAME/$PASSWORD \
    neo4j
