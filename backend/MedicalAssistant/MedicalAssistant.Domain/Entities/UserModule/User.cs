using MedicalAssistant.Domain.Entities;
using System.ComponentModel.DataAnnotations.Schema;
namespace MedicalAssistant.Domain.Entities.UserModule;

public class User : BaseEntity
{
    public string FullName { get; set; } = string.Empty;
    public string Email { get; set; } = string.Empty;
    [Column("PasswordHash")]
    public string PasswordHash { get; set; } = string.Empty;
    public string Role { get; set; } = string.Empty; // Patient | Doctor | Admin
    public string? PhoneNumber { get; set; }
    public bool IsActive { get; set; } = true;
}