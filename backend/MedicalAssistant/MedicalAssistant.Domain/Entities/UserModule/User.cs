using System.ComponentModel.DataAnnotations.Schema;

namespace MedicalAssistant.Domain.Entities.UserModule;

public class User : BaseEntity
{
    public string FullName { get; set; } = string.Empty; // [cite: 79]
    public string Email { get; set; } = string.Empty; // [cite: 79]

    [Column("PasswordHash")]
    public string PasswordHash { get; set; } = string.Empty; // [cite: 79]

    public string Role { get; set; } = string.Empty; // Patient | Doctor | Admin [cite: 79]
    public string? PhoneNumber { get; set; } // [cite: 79]
    public DateTime? BirthDate { get; set; } // Birthdate for patients
    public bool IsActive { get; set; } = true; // [cite: 79]
    public bool IsDeleted { get; set; } = false; // 
    public DateTime CreatedAt { get; set; } = DateTime.UtcNow; // 
    public DateTime? UpdatedAt { get; set; } // 
}