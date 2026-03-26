using MedicalAssistant.Domain.Entities.UserModule;

namespace MedicalAssistant.Domain.Entities.AdminModule;

public class Admin : User
{
    
    public DateTime? LastLoginAt { get; set; }
}