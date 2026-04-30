# Stage 1: Build the application
FROM mcr.microsoft.com/dotnet/sdk:8.0 AS build
WORKDIR /source

# Copy all .csproj files and restore dependencies
COPY ./*.sln .
COPY */*.csproj ./
RUN for file in $(ls ./*.csproj); do mkdir -p ${file%.*} && mv $file ${file%.*}; done
RUN dotnet restore

# Copy the entire source code
COPY . .
WORKDIR /source/MedicalAssistant.Web

# Publish the web project
RUN dotnet publish -c Release -o /app/publish --no-restore

# Stage 2: Create the final runtime image
FROM mcr.microsoft.com/dotnet/aspnet:8.0
WORKDIR /app
COPY --from=build /app/publish .
ENTRYPOINT ["dotnet", "MedicalAssistant.Web.dll"]
