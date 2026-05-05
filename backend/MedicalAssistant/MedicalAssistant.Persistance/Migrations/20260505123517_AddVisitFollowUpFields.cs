using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace MedicalAssistant.Persistance.Migrations
{
    /// <inheritdoc />
    public partial class AddVisitFollowUpFields : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.AddColumn<int>(
                name: "FollowUpAfterDays",
                table: "PatientVisits",
                type: "int",
                nullable: true);

            migrationBuilder.AddColumn<string>(
                name: "FollowUpNotes",
                table: "PatientVisits",
                type: "text",
                nullable: true);

            migrationBuilder.AddColumn<bool>(
                name: "FollowUpRequired",
                table: "PatientVisits",
                type: "bit",
                nullable: true);
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropColumn(
                name: "FollowUpAfterDays",
                table: "PatientVisits");

            migrationBuilder.DropColumn(
                name: "FollowUpNotes",
                table: "PatientVisits");

            migrationBuilder.DropColumn(
                name: "FollowUpRequired",
                table: "PatientVisits");
        }
    }
}
