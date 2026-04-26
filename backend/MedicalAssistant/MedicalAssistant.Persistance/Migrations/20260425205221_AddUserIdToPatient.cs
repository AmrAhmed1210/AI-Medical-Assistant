using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace MedicalAssistant.Persistance.Migrations
{
    /// <inheritdoc />
    public partial class AddUserIdToPatient : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            // Use compatible types for both SQL Server and PostgreSQL
            var isPostgres = migrationBuilder.ActiveProvider == "Npgsql.EntityFrameworkCore.PostgreSQL";
            var stringType = isPostgres ? "text" : "nvarchar(max)";
            var intType = isPostgres ? "integer" : "int";

            migrationBuilder.AddColumn<string>(
                name: "Type",
                table: "Session",
                type: stringType,
                nullable: false,
                defaultValue: "");

            migrationBuilder.AddColumn<int>(
                name: "UserId",
                table: "Patients",
                type: intType,
                nullable: true);

            migrationBuilder.AddColumn<string>(
                name: "AttachmentUrl",
                table: "Message",
                type: stringType,
                nullable: true);

            migrationBuilder.AddColumn<string>(
                name: "FileName",
                table: "Message",
                type: stringType,
                nullable: true);

            migrationBuilder.AddColumn<string>(
                name: "MessageType",
                table: "Message",
                type: stringType,
                nullable: false,
                defaultValue: "");

            migrationBuilder.AddColumn<string>(
                name: "PhotoUrl",
                table: "DoctorApplications",
                type: stringType,
                nullable: true);

            migrationBuilder.CreateIndex(
                name: "IX_Patients_UserId",
                table: "Patients",
                column: "UserId");

            migrationBuilder.AddForeignKey(
                name: "FK_Patients_Users_UserId",
                table: "Patients",
                column: "UserId",
                principalTable: "Users",
                principalColumn: "Id");
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropForeignKey(
                name: "FK_Patients_Users_UserId",
                table: "Patients");

            migrationBuilder.DropIndex(
                name: "IX_Patients_UserId",
                table: "Patients");

            migrationBuilder.DropColumn(
                name: "Type",
                table: "Session");

            migrationBuilder.DropColumn(
                name: "UserId",
                table: "Patients");

            migrationBuilder.DropColumn(
                name: "AttachmentUrl",
                table: "Message");

            migrationBuilder.DropColumn(
                name: "FileName",
                table: "Message");

            migrationBuilder.DropColumn(
                name: "MessageType",
                table: "Message");

            migrationBuilder.DropColumn(
                name: "PhotoUrl",
                table: "DoctorApplications");
        }
    }
}
