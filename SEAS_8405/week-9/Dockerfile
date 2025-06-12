# --- Stage 1: Build with Maven ---
FROM maven:3.8.5-openjdk-11 AS build
WORKDIR /app
COPY . .
RUN mvn clean package -DskipTests

# --- Stage 2: Runtime ---
FROM openjdk:11-jre-slim
WORKDIR /app

# Copy JAR and log4j2.xml config file
COPY --from=build /app/target/log4shell-demo-0.0.1-SNAPSHOT.jar /app.jar
COPY src/main/resources/log4j2.xml /app/log4j2.xml

# Force Log4j2 to load the config
ENTRYPOINT ["java", "-Dlog4j.configurationFile=log4j2.xml", "-jar", "/app.jar"]
