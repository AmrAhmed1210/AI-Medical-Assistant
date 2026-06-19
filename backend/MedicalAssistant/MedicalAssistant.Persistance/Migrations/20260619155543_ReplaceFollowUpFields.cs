using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace MedicalAssistant.Persistance.Migrations
{
    /// <inheritdoc />
    public partial class ReplaceFollowUpFields : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropColumn(
                name: "FollowUpAfterDays",
                table: "PatientVisits");

            migrationBuilder.AddColumn<string>(
                name: "FollowUpDate",
                table: "PatientVisits",
                type: "nvarchar(max)",
                nullable: true);

            migrationBuilder.AddColumn<string>(
                name: "FollowUpTime",
                table: "PatientVisits",
                type: "nvarchar(max)",
                nullable: true);
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropColumn(
                name: "FollowUpDate",
                table: "PatientVisits");

            migrationBuilder.DropColumn(
                name: "FollowUpTime",
                table: "PatientVisits");

            migrationBuilder.AddColumn<int>(
                name: "FollowUpAfterDays",
                table: "PatientVisits",
                type: "int",
                nullable: true);
        }
    }
}
