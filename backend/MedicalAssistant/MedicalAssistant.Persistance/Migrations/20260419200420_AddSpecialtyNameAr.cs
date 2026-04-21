using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace MedicalAssistant.Persistance.Migrations
{
    /// <inheritdoc />
    public partial class AddSpecialtyNameAr : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.Sql(@"
                IF COL_LENGTH('Specialties', 'NameAr') IS NULL
                BEGIN
                    ALTER TABLE [Specialties] ADD [NameAr] nvarchar(max) NULL;
                END
            ");

            migrationBuilder.Sql(@"
                IF COL_LENGTH('Doctors', 'UserId') IS NULL
                BEGIN
                    ALTER TABLE [Doctors] ADD [UserId] int NULL;
                END
            ");

            migrationBuilder.Sql(@"
                IF NOT EXISTS (
                    SELECT 1
                    FROM sys.indexes
                    WHERE name = 'IX_Doctors_UserId'
                      AND object_id = OBJECT_ID('Doctors')
                )
                BEGIN
                    CREATE INDEX [IX_Doctors_UserId] ON [Doctors]([UserId]);
                END
            ");

            migrationBuilder.Sql(@"
                IF NOT EXISTS (
                    SELECT 1
                    FROM sys.foreign_keys
                    WHERE name = 'FK_Doctors_Users_UserId'
                )
                BEGIN
                    ALTER TABLE [Doctors]
                    ADD CONSTRAINT [FK_Doctors_Users_UserId]
                    FOREIGN KEY ([UserId]) REFERENCES [Users]([Id]) ON DELETE SET NULL;
                END
            ");
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.Sql(@"
                IF EXISTS (
                    SELECT 1
                    FROM sys.foreign_keys
                    WHERE name = 'FK_Doctors_Users_UserId'
                )
                BEGIN
                    ALTER TABLE [Doctors] DROP CONSTRAINT [FK_Doctors_Users_UserId];
                END
            ");

            migrationBuilder.Sql(@"
                IF EXISTS (
                    SELECT 1
                    FROM sys.indexes
                    WHERE name = 'IX_Doctors_UserId'
                      AND object_id = OBJECT_ID('Doctors')
                )
                BEGIN
                    DROP INDEX [IX_Doctors_UserId] ON [Doctors];
                END
            ");

            migrationBuilder.Sql(@"
                IF COL_LENGTH('Doctors', 'UserId') IS NOT NULL
                BEGIN
                    ALTER TABLE [Doctors] DROP COLUMN [UserId];
                END
            ");

            migrationBuilder.Sql(@"
                IF COL_LENGTH('Specialties', 'NameAr') IS NOT NULL
                BEGIN
                    ALTER TABLE [Specialties] DROP COLUMN [NameAr];
                END
            ");
        }
    }
}
